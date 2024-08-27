from copy import copy
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jax.typing import ArrayLike

from msmjax.bspline_interpolation.coefficients import (
    compute_coeffs_with_truncation,
)
from msmjax.convenience import set_up_kernels_and_grids
from msmjax.gridops_multidim import (
    create_compute_f_oneplus_via_potential,
    create_compute_U_oneplus_direct,
)


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
        points = onp.array([onp.cos(phis), onp.sin(phis)]).T
    elif n_dim == 3:
        phis = onp.linspace(0, 2 * onp.pi, 200)
        thetas = onp.linspace(0, onp.pi, 200)
        phis, thetas = onp.meshgrid(phis, thetas)
        phis = phis.ravel()
        thetas = thetas.ravel()
        points = onp.array(
            [
                onp.cos(phis) * onp.sin(thetas),
                onp.sin(phis) * onp.sin(thetas),
                onp.cos(thetas),
            ]
        ).T
    else:
        raise ValueError

    points_at_cutoff = cutoff * points
    points_at_cutoff_transformed = points_at_cutoff @ inverse

    # TODO: Are these transformed spacings correct?
    single_grid_cell = (
        cell
        / onp.linalg.norm(cell, axis=1)  # TODO: does this broadcast correctly?
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


def _onedim_convolution_fn(in1: ArrayLike, in2: ArrayLike):
    # TODO: method="fft"?
    return jax.scipy.signal.convolve(in1, in2, mode="same")


def _compute_kernel_stencil(values: ArrayLike, omega: ArrayLike):
    convolved = values
    # TODO: Is this sequential application of 1d convolutions the fastest thing
    #  one can do?
    # TODO: exploit symmetry?
    for i in range(values.ndim):
        convolved = jnp.apply_along_axis(
            func1d=_onedim_convolution_fn, axis=i, arr=convolved, in2=omega
        )
    return convolved


def _construct_all_kernel_stencils(
    kernel_fns: List[Callable],  # TODO: appropriate type hint?
    omega,
    points,  # TODO: pass points or directly the distances?
    includes_toplevel: bool,
    points_toplevel,  # TODO: pass points or directly the distances?
):
    # TODO: raise error when `includes_toplevel=True`, but sizes not given
    n_levels = len(kernel_fns)

    # Level zero (where there is no grid)
    stencils = [None]

    # Level one
    distances = jnp.linalg.norm(points, axis=-1)
    fn_vals_at_points = kernel_fns[1](distances)
    stencils.append(_compute_kernel_stencil(fn_vals_at_points, omega))

    # Intermediate levels:
    # For the type of kernel splitting used here, the intermediate-level
    # kernel values (and thus stencils) can be computed by simply dividing
    # the one from the previous level by 2. But this need not hold for
    # other kernels or ways of splitting!
    # TODO: can this be done by a broadcast multiplication?
    for lvl in range(2, n_levels - 1):
        stencils.append(0.5 * stencils[-1])

    # Highest included level
    if includes_toplevel:
        # TODO: Some possible efficiency gain by precomputing `points_cartesian`
        #  or `points_cartesian_toplevel`, whichever is larger in shape,
        #  and then getting the smaller by indexing into the larger
        # TODO: For the size of the top level stencil chosen sufficiently
        #  large (I think it needs to be the grid size + half the length of
        #  omega as padding), constructing it is very costly
        distances_toplevel = 2 ** (n_levels - 2) * jnp.linalg.norm(
            points_toplevel, axis=-1
        )
        fn_vals_at_points_toplevel = kernel_fns[-1](distances_toplevel)
        stencils.append(
            _compute_kernel_stencil(fn_vals_at_points_toplevel, omega)
        )
    else:
        stencils.append(0.5 * stencils[-1])

    return stencils


# TODO: "dynamic" in name?
def make_kernel_stencil_construction_fn(
    kernel_fns: List[Callable],
    sizes_from_center: Sequence[int],
    reference_cell,
    reference_spacings,
    omega,
    includes_toplevel: bool,
    sizes_from_center_toplevel=None,
):
    # TODO: raise error when `includes_toplevel=True`, but sizes not given

    reference_side_lengths = onp.linalg.norm(reference_cell, axis=1)
    reference_spacings = onp.asarray(reference_spacings)
    spacings_unitcube = reference_spacings / reference_side_lengths

    indices_1d = [onp.arange(-s, s + 1) for s in sizes_from_center]
    indices = onp.stack(onp.meshgrid(*indices_1d, indexing="ij"), axis=-1)
    points_unitcube = indices * spacings_unitcube
    if includes_toplevel:
        indices_1d_toplevel = [
            onp.arange(-s, s + 1) for s in sizes_from_center_toplevel
        ]
        indices_toplevel = onp.stack(
            onp.meshgrid(*indices_1d_toplevel, indexing="ij"), axis=-1
        )
        points_unitcube_toplevel = indices_toplevel * spacings_unitcube

    def construct_kernel_stencils(cell):
        return _construct_all_kernel_stencils(
            kernel_fns=kernel_fns,
            omega=omega,
            points=points_unitcube @ cell,
            includes_toplevel=includes_toplevel,
            points_toplevel=(
                points_unitcube_toplevel @ cell if includes_toplevel else None
            ),
        )

    return construct_kernel_stencils


def set_up_grids_unitcube(
    box_lengths_original, pbc, msm_params_original: dict
):
    # TODO: Change argument to `cell` instead of `box_lengths`
    # TODO: Pass msm_params as dict or as the individual parameters? (What
    #  is more consistent with other functions? Probably the latter)
    box_lengths_unitcube = onp.ones(len(pbc))
    msm_params_unitcube = copy(msm_params_original)
    msm_params_unitcube["level_one_gridspacing"] = (
        msm_params_original["level_one_gridspacing"] / box_lengths_original[0]
    )
    _, grids = set_up_kernels_and_grids(
        box_lengths=box_lengths_unitcube, pbcs=pbc, **msm_params_unitcube
    )

    return grids


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
        includes_toplevel=includes_toplevel,
        sizes_from_center_toplevel=sizes_toplevel,
    )

    if cell_mode == "ortho":
        to_unit_cube = lambda positions, cell: positions / jnp.diag(cell)
        backtransform_forces = lambda f, cell: f / jnp.diag(cell)
    elif cell_mode == "general":
        to_unit_cube = lambda positions, cell: positions @ jnp.linalg.pinv(
            cell
        )
        # TODO: why do we need to transpose here?
        backtransform_forces = lambda f, cell: f @ jnp.linalg.pinv(cell).T
    else:
        raise ValueError("Invalid `cell_mode`.")

    # TODO: Allow choosing different setup functions for U_oneplus (`create_compute_U_oneplus_via_potential`)
    #  And what about the same choice for forces?
    compute_U1plus_unitcube_direct = create_compute_U_oneplus_direct(
        grids=grids_unit_cube, convolution_methods=convolution_methods
    )
    compute_f1plus_unitcube_via_potential = (
        create_compute_f_oneplus_via_potential(
            grids=grids_unit_cube, convolution_methods=convolution_methods
        )
    )

    def compute_U1plus_flex_cell(positions, charges, cell):
        kernel_stencils = construct_kernel_stencils(cell)
        return compute_U1plus_unitcube_direct(
            to_unit_cube(positions, cell), charges, kernel_stencils
        )

    def compute_f1plus_flex_cell(positions, charges, cell):
        kernel_stencils = construct_kernel_stencils(cell)
        f = compute_f1plus_unitcube_via_potential(
            to_unit_cube(positions, cell), charges, kernel_stencils
        )
        # TODO: How to correctly transform back the forces from the unit cube?
        backtransform_forces(f, cell)

    if forces:
        return compute_U1plus_flex_cell, compute_f1plus_flex_cell
    else:
        return compute_U1plus_flex_cell
