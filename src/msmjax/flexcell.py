from copy import copy
from typing import Callable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from msmfornn.splines.coefficients import compute_coeffs_withtruncation

from msmjax.convenience import set_up_grids_and_kernels
from msmjax.gridops_multidim import create_compute_U_oneplus_direct


def onedim_convolution_fn(in1, in2):
    return jax.scipy.signal.convolve(in1, in2, mode="same")


def compute_kernel_stencil(values, omega):
    convolved = values
    for i in range(values.ndim):
        convolved = jnp.apply_along_axis(
            func1d=onedim_convolution_fn, axis=i, arr=convolved, in2=omega
        )
    return convolved


def make_kernel_stencil_construction_fn(
    kernel_fns: List[Union[None, Callable]],  # TODO: correct type hint?
    sizes_from_center: Sequence[int],
    reference_cell,
    reference_spacings,
    omega,
    includes_toplevel,
    sizes_from_center_toplevel=None,
):
    # TODO: raise error when `includes_toplevel=True`, but sizes not given

    ref_side_lengths = onp.linalg.norm(reference_cell, axis=1)
    reference_spacings = onp.asarray(reference_spacings)
    spacings_unitcube = reference_spacings / ref_side_lengths

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
        stencils = [None]

        # Level one
        points_cartesian = points_unitcube @ cell
        distances_cartesian = jnp.linalg.norm(points_cartesian, axis=-1)
        fn_vals_at_points = kernel_fns[1](distances_cartesian)
        stencils.append(compute_kernel_stencil(fn_vals_at_points, omega))

        # Intermediate levels
        for lvl in range(2, len(kernel_fns) - 1):
            stencils.append(0.5 * stencils[-1])

        # Highest included level
        if includes_toplevel:
            # TODO: Some possible efficiency gain by precomputing `points_cartesian`
            #  or `points_cartesian_toplevel`, whichever is larger in shape,
            #  and then getting the smaller by indexing into the larger
            # TODO: For the size of the top level stencil chosen sufficiently
            #  large (I think it needs to be the grid size + half the length of
            #  omega as padding), constructing it is very costly
            points_cartesian_toplevel = points_unitcube_toplevel @ cell
            distances_cartesian_toplevel = 2 ** (
                len(kernel_fns) - 2
            ) * jnp.linalg.norm(points_cartesian_toplevel, axis=-1)
            fn_vals_at_points_toplevel = kernel_fns[-1](
                distances_cartesian_toplevel
            )
            stencils.append(
                compute_kernel_stencil(fn_vals_at_points_toplevel, omega)
            )
        else:
            stencils.append(0.5 * stencils[-1])

        return stencils

    return construct_kernel_stencils


def set_up_grids_unitcube(
    box_lengths_original, pbc, msm_params_original: dict
):
    # TODO: Pass msm_params as dict or as the individual parameters? (What
    #  be more consistent with other functions? Probably the latter)
    box_lengths_unitcube = onp.ones(len(pbc))
    msm_params_unitcube = copy(msm_params_original)
    msm_params_unitcube["level_one_gridspacing"] = (
        msm_params_original["level_one_gridspacing"] / box_lengths_original[0]
    )
    _, grids, _ = set_up_grids_and_kernels(
        box_lengths=box_lengths_unitcube, pbcs=pbc, **msm_params_unitcube
    )

    return grids


def make_flex_cell_U1plus_fn(
    kernel_fns: List[Union[None, Callable]],  # TODO: correct type hint?
    pbc,
    reference_cell,
    level_one_gridspacing,  # TODO: "reference" in the name?
    level_zero_cutoff,
    p,
    mu,
    n_levels,
    max_compression_factor: float = 1.25,  # TODO: variable name, default value?
    convolution_methods=None,
):
    # TODO: Allow specifying stencil sizes explicitly as well as via
    #  `max_compression_factor`?

    n_dim = len(pbc)
    pbc = onp.asarray(pbc)
    # TODO: different spacings along different directions
    reference_spacings = onp.array([level_one_gridspacing] * n_dim)

    # TODO: While I'm at it, move this function from msmfornn to msmjax
    omega, _ = compute_coeffs_withtruncation(p=p, mu=mu)

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

    if onp.all(pbc):
        includes_toplevel = False
        sizes_toplevel = None
    elif onp.all(~pbc):
        includes_toplevel = True
        padding = len(omega) // 2
        sizes_toplevel = tuple(s + padding for s in grids_unit_cube[-1].shape)
    else:
        raise ValueError("Mixed boundary conditions not supported yet")

    cutoff_lvl_1 = 2 * level_zero_cutoff
    # TODO: variable name
    n_points_cutoff_oneside = onp.ceil(cutoff_lvl_1 / level_one_gridspacing)
    stencil_sizes_from_center = [
        int(onp.ceil(max_compression_factor * n_points_cutoff_oneside))
    ] * n_dim

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
    # TODO: remove jit from here, once the whole function has been made jittable
    construct_kernel_stencils = jax.jit(construct_kernel_stencils)

    def to_unit_cube(positions, cell):
        # TODO: version for general parallelepipeds with pinv
        return positions / jnp.diag(cell)

    # TODO: Allow choosing different setup functions for U_oneplus (`create_compute_U_oneplus_via_potential`)
    #  And what about the same choice for forces?
    # TODO: Remove `kernel_stencils` parameter from
    #  `create_compute_U_oneplus_direct` and instead make a parameter of its
    #  returned function
    compute_U1plus_unit_cube = create_compute_U_oneplus_direct(
        grids=grids_unit_cube, convolution_methods=convolution_methods
    )

    def compute_U1plus_flex_cell(positions, charges, cell):
        kernel_stencils = construct_kernel_stencils(cell)
        return compute_U1plus_unit_cube(
            to_unit_cube(positions, cell), charges, kernel_stencils
        )

    return compute_U1plus_flex_cell
