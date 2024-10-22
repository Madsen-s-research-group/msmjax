from copy import copy
from typing import Callable, List, Literal, Optional, Sequence, Union

import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt

from msmjax.bspline_interpolation.coefficients import (
    compute_coeffs_with_truncation,
)
from msmjax.convenience import set_up_kernels_and_grids
from msmjax.gridops_multidim import (
    create_compute_f_oneplus_via_potential,
    create_compute_U_and_f_oneplus_via_potential,
    create_compute_U_oneplus_direct,
)
from msmjax.kernels import make_kernel_stencil_construction_fn


def determine_min_kernel_stencil_size(cell, spacings, cutoff):
    # TODO: Should the parameter names for spacings and r_cut suggest one
    #  specific grid level? In principle, if they're given at the same level,
    #  it does not matter which, since both are doubled at each level.
    #  But OTOH, the risk of inadvertently passing the level-ONE spacing and
    #  together with the level-ZERO cutoff should be minimized

    # TODO: unit test this function

    n_dim = cell.shape[0]
    inverse = onp.linalg.inv(cell)

    # TODO: Can this be made more generic (same code working for all dimensions)?
    # TODO: Could this be implemented via the usual max-cutoff formula
    #  (as implemented in `get_max_cutoff_3d`) instead? (Keep enlarging the
    #  cell passed to `get_max_cutoff_3d` in discrete steps, corresponding to
    #  adding an additional grid point, until the cutoff fits)
    if n_dim == 1:
        return tuple([onp.floor(cutoff / spacings).astype(int)])
    elif n_dim == 2:
        phis = onp.linspace(0, 2 * onp.pi, 500)
        points_unitsphere = onp.array([onp.cos(phis), onp.sin(phis)]).T
    elif n_dim == 3:
        phis = onp.linspace(0, 2 * onp.pi, 200)
        thetas = onp.linspace(0, onp.pi, 200)
        phis, thetas = onp.meshgrid(phis, thetas)
        phis = phis.ravel()
        thetas = thetas.ravel()
        points_unitsphere = onp.array(
            [
                onp.cos(phis) * onp.sin(thetas),
                onp.sin(phis) * onp.sin(thetas),
                onp.cos(thetas),
            ]
        ).T
    else:
        raise ValueError("Spatial dimensions greater than 3 not supported.")

    points_at_cutoff = cutoff * points_unitsphere
    points_at_cutoff_transformed = points_at_cutoff @ inverse

    single_grid_cell = (
        cell
        / onp.linalg.norm(cell, axis=1)[:, onp.newaxis]
        * onp.atleast_1d(spacings)[:, onp.newaxis]
    )
    spacings_transformed = onp.diag(single_grid_cell @ inverse)

    # The maximum taken from the precomputed cutoff sphere points may be
    # slightly too low, because they incompletely sample the cutoff sphere
    # => use an additional small tolerance
    tol = 1.0e-3
    sizes_from_center = onp.floor(
        points_at_cutoff_transformed.max(axis=0) / spacings_transformed + tol
    ).astype(int)

    return tuple(sizes_from_center)


def set_up_grids_unitcube(
    box_lengths_original, pbc, msm_params_original: dict
):
    # TODO: Change argument to `cell` instead of `box_lengths`
    # TODO: Pass msm_params as dict or as the individual parameters? (What
    #  is more consistent with other functions? Probably the latter)
    box_lengths_unitcube = onp.ones(len(pbc))
    msm_params_unitcube = copy(msm_params_original)
    msm_params_unitcube["level_one_gridspacing"] = (
        msm_params_original["level_one_gridspacing"] / box_lengths_original
    )
    _, grids = set_up_kernels_and_grids(
        box_lengths=box_lengths_unitcube, pbcs=pbc, **msm_params_unitcube
    )

    return grids


def make_unitcube_transform_fns_ortho(cell):
    inverse = 1.0 / jnp.diag(cell)
    transform_pos = lambda x: x * inverse
    backtransform_grad = lambda x: x * inverse
    return transform_pos, backtransform_grad


def make_unitcube_transform_fns_general(cell):
    inverse = jnp.linalg.pinv(cell)
    transform_pos = lambda x: x @ inverse
    backtransform_grad = lambda dx: dx @ inverse.T
    return transform_pos, backtransform_grad


def make_flex_cell_U1plus_fn(
    kernel_fns: List[Callable],
    pbc: Sequence[bool],
    reference_cell: npt.ArrayLike,
    level_one_gridspacing,  # TODO: "reference" in the name?
    level_zero_cutoff,
    p: int,
    mu: int,
    n_levels: int,
    convolution_methods=None,
    forces: bool = False,
    stencil_padding: Optional[Union[int, Sequence[int]]] = None,
    cell_mode: Literal["ortho", "general"] = "ortho",
):
    # TODO: Option to return auxiliary information, like the grids, stencils,
    #  omega, ...?

    # TODO: Offer passing both `stencil_padding` and `stencil_size`
    #  (or `compression_tolerance_factor`)? (not both at the same time)

    # TODO: In addition to explicit stencil padding, allow specification via
    #  (something like) `max_compression_factor` as well? (maybe more intuitive)

    n_dim = len(pbc)
    pbc = onp.asarray(pbc)
    if onp.isscalar(level_one_gridspacing):
        reference_spacings = onp.full(n_dim, level_one_gridspacing)
    else:
        reference_spacings = onp.asarray(level_one_gridspacing)

    if stencil_padding is None:
        stencil_padding = 0

    omega, _ = compute_coeffs_with_truncation(p=p, mu=mu)
    grids_unit_cube = set_up_grids_unitcube(
        box_lengths_original=onp.linalg.norm(reference_cell, axis=1),
        pbc=pbc,
        msm_params_original=dict(
            level_one_gridspacing=level_one_gridspacing,
            level_zero_cutoff=level_zero_cutoff,
            p=p,
            mu=mu,
            n_levels=n_levels,
        ),
    )

    cutoff_lvl_1 = 2 * level_zero_cutoff
    stencil_sizes_from_center = determine_min_kernel_stencil_size(
        cell=reference_cell,
        spacings=level_one_gridspacing,
        cutoff=cutoff_lvl_1,
    )

    if onp.all(pbc):
        includes_toplevel = False
        sizes_toplevel = None
    elif onp.all(~pbc):
        includes_toplevel = True
        # For no information loss during convolution, we need to add half the
        # length of omega in padding.
        sizes_toplevel = tuple(
            s + len(omega) // 2 for s in grids_unit_cube[-1].shape
        )
        sizes_toplevel = tuple(onp.array(sizes_toplevel) + stencil_padding)
    else:
        raise ValueError("Mixed boundary conditions not supported yet")

    stencil_sizes_from_center = tuple(
        onp.array(stencil_sizes_from_center) + stencil_padding
    )

    # TODO: trim unnecessarily large stencils (especially: top level for non-periodic)
    construct_kernel_stencils = make_kernel_stencil_construction_fn(
        kernel_fns=kernel_fns,
        sizes_from_center=stencil_sizes_from_center,
        reference_cell=reference_cell,
        reference_spacings=reference_spacings,
        omega=omega,
        kernels_include_toplevel=includes_toplevel,
        sizes_from_center_toplevel=sizes_toplevel,
    )

    if cell_mode == "ortho":
        make_unitcube_transform_fns = make_unitcube_transform_fns_ortho
    elif cell_mode == "general":
        make_unitcube_transform_fns = make_unitcube_transform_fns_general
    else:
        raise ValueError("Invalid `cell_mode`.")

    compute_U1plus_unitcube = create_compute_U_oneplus_direct(
        grids=grids_unit_cube, convolution_methods=convolution_methods
    )
    compute_f1plus_unitcube = create_compute_f_oneplus_via_potential(
        grids=grids_unit_cube, convolution_methods=convolution_methods
    )
    compute_U1plus_and_f1plus_unitcube = (
        create_compute_U_and_f_oneplus_via_potential(
            grids=grids_unit_cube, convolution_methods=convolution_methods
        )
    )

    # TODO: The following functions could perhaps be implemented via a general
    #  wrapper that takes a "construct_kernel_stencils" fn, a (tuple of?)
    #  "transform" fn(s), and a "compute" fn

    def compute_U1plus_flexcell(positions, charges, cell):
        transform_pos, backtransform_grad = make_unitcube_transform_fns(cell)
        return compute_U1plus_unitcube(
            transform_pos(positions), charges, construct_kernel_stencils(cell)
        )

    def compute_f1plus_flexcell(positions, charges, cell):
        transform_pos, backtransform_grad = make_unitcube_transform_fns(cell)
        f = compute_f1plus_unitcube(
            transform_pos(positions), charges, construct_kernel_stencils(cell)
        )
        return backtransform_grad(f)

    def compute_U1plus_and_f1plus_flexcell(positions, charges, cell):
        transform_pos, backtransform_grad = make_unitcube_transform_fns(cell)
        e, f = compute_U1plus_and_f1plus_unitcube(
            transform_pos(positions), charges, construct_kernel_stencils(cell)
        )
        return e, backtransform_grad(f)

    if forces:
        return (
            compute_U1plus_flexcell,
            compute_f1plus_flexcell,
            compute_U1plus_and_f1plus_flexcell,
        )
    else:
        return compute_U1plus_flexcell
