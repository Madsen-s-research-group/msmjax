import numpy as onp
from jax import numpy as jnp
from msmfornn.grid_to_grid_mapping import (
    compute_kernel_stencils_all_gridlevels,
)
from msmfornn.gridtools import construct_grids_all_levels
from msmfornn.splines.coefficients import compute_coeffs_withtruncation


def _construct_kernel_stencils(
    kernels,
    box_lengths,
    level_one_gridspacing,
    level_zero_cutoff,
    n_levels,
    p,
    mu,
):
    omega, _ = compute_coeffs_withtruncation(p=p, mu=mu)
    omega_zeroplus = omega[len(omega) // 2 :]
    grids_oldmsm = construct_grids_all_levels(
        min_pos=onp.zeros_like(box_lengths),
        max_pos=box_lengths,
        p=p,
        level_one_gridspacing=level_one_gridspacing,
        max_gridlevel=n_levels,
    )
    kernel_stencils_nonnegative = compute_kernel_stencils_all_gridlevels(
        kernelfunctions=kernels,
        grids=grids_oldmsm,
        level_zero_cutoff=level_zero_cutoff,
        omega_zeroplus=omega_zeroplus,
    )
    kernel_stencils = [None]
    for stncl in kernel_stencils_nonnegative[1:]:
        pw = [(s - 1, 0) for s in stncl.shape]
        stncl_symm = jnp.pad(stncl, pad_width=pw, mode="reflect")
        kernel_stencils.append(stncl_symm)

    return kernel_stencils
