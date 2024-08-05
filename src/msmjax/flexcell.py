from copy import copy
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as onp

from msmjax.convenience import set_up_grids_and_kernels
from msmjax.gridops_multidim import create_compute_U_oneplus_direct


def onedim_convolution_fn(a):
    return jax.scipy.signal.convolve(a, in2=omega, mode="same")


def compute_kernel_stencil(values):
    convolved = values
    for i in range(values.ndim):
        convolved = jnp.apply_along_axis(
            func1d=onedim_convolution_fn, axis=i, arr=convolved
        )
    return convolved


def make_kernel_stencil_construction_fn(
    kernel_fns: List[Callable],
    stencil_sizes_from_center: Sequence[int],
    ref_cell,
    ref_spacings,
    includes_toplevel=True,
    sizes_toplevel=None,  # TODO: name
):
    # TODO: raise error when `includes_toplevel=True`, but sizes not given

    ref_side_lengths = onp.linalg.norm(ref_cell, axis=1)
    ref_spacings = onp.asarray(ref_spacings)
    spacings_unitcube = ref_spacings / ref_side_lengths

    indices_1d = [onp.arange(-s, s + 1) for s in stencil_sizes_from_center]
    indices = onp.stack(onp.meshgrid(*indices_1d, indexing="ij"), axis=-1)
    points_unitcube = indices * spacings_unitcube
    if includes_toplevel:
        indices_1d_toplevel = [onp.arange(-s, s + 1) for s in sizes_toplevel]
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
        stencils.append(compute_kernel_stencil(fn_vals_at_points))

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
            stencils.append(compute_kernel_stencil(fn_vals_at_points_toplevel))
        else:
            stencils.append(0.5 * stencils[-1])

        return stencils

    return construct_kernel_stencils


def set_up_grids_unitcube(box_lengths_original, pbc, msm_params_original):
    box_lengths_unitcube = onp.ones(len(pbc))
    msm_params_unitcube = copy(msm_params_original)
    msm_params_unitcube["level_one_gridspacing"] = (
        msm_params_original["level_one_gridspacing"] / box_lengths_original[0]
    )
    _, grids, _ = set_up_grids_and_kernels(
        box_lengths=box_lengths_unitcube, pbcs=pbc, **msm_params_unitcube
    )

    return grids


def wrapped_compute_U0_flexcell(positions, charges, cell):
    def to_unitcube(pos):
        # TODO: general parallelepipeds
        return pos / onp.diag(cell)

    # TODO: unnecessarily large stencils could be trimmed
    compute_U0_unitcube = create_compute_U_oneplus_direct(
        grids=grids_unitcube,
        kernel_stencils=construct_kernel_stencils_dynamic(cell),
    )
    compute_U0_unitcube = jax.jit(compute_U0_unitcube)

    return compute_U0_unitcube(to_unitcube(positions), charges)
