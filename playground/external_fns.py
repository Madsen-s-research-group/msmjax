import numpy as onp
from jax.typing import ArrayLike

from msmjax.bspline_interpolation.gridops_clean import set_up_grids_all_levels


def make_clean_prolongate_1d(
    axis_source_coarse: BSplineInterpolationAxis,
    axis_target_fine: BSplineInterpolationAxis,
):
    # TODO: Change function parameters to (n_points_in, n_points_out, p, periodic)
    #  => directly compute J from p during setup, this should be fine
    p = axis_source_coarse.p
    J_zeroplus = axis_source_coarse.J_zeroplus
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    n_points_in = axis_source_coarse.n_total
    n_points_out = axis_target_fine.n_total
    # TODO: check n_points_in >= n_points_out?
    periodic = axis_source_coarse.periodic

    # TODO: External factory function that creates both `zero_align_idx` and
    #  `to_positional_idx` from `periodic` and `p`

    def zero_align_idx(positional_idx):
        if periodic:
            return positional_idx
        else:
            return positional_idx - p // 2

    def to_positional_idx(zero_aligned_idx):
        if periodic:
            return zero_aligned_idx
        else:
            return zero_aligned_idx + p // 2


def construct_stencils(
    omega: ArrayLike,
    n_levels_intermediate: int,
    include_toplevel: bool,
    k_lvl_1: Callable[[ArrayLike], Array] = None,
    sizes_intermediate: tuple[int, ...] = None,
    spacings_or_gridcell_lvl_1: ArrayLike = None,
    k_toplevel: Callable[[ArrayLike], Array] = None,
    grid_shape_toplevel: tuple[int, ...] = None,
) -> list[Array]:
    # Placeholder for level zero (l = 0), at which there is no grid:
    stencils = [None]

    if n_levels_intermediate < 0:
        raise ValueError("n_levels_intermediate must be >= 0")
    if n_levels_intermediate == 0 and not include_toplevel:
        raise ValueError(
            "n_levels_intermediate = 0 and include_toplevel = False "
            "at the same is not allowed (this would mean that "
            "there isn't a single grid level)."
        )
    args_intermediate = [
        k_lvl_1,
        sizes_intermediate,
        spacings_or_gridcell_lvl_1,
    ]
    if n_levels_intermediate > 0 and any(
        [x is None for x in args_intermediate]
    ):
        raise ValueError(
            "k_lvl_1, sizes_intermediate, spacings_or_gridcell_lvl_1 "
            "are required when n_levels_intermediate > 0."
        )
    args_toplevel = [k_toplevel, grid_shape_toplevel]
    if include_toplevel and any([x is None for x in args_toplevel]):
        raise ValueError(
            "k_toplevel, grid_shape_toplevel "
            "are required when include_toplevel = True."
        )

    # Intermediate levels (l = 1 ... L - 1):
    if n_levels_intermediate > 0:
        distances_lvl_1 = get_distances(
            sizes_intermediate, spacings_or_gridcell_lvl_1
        )
        kernel_values_at_gridpoints = k_lvl_1(distances_lvl_1)
        stencils.append(
            _compute_kernel_stencil(kernel_values_at_gridpoints, omega)
        )
        for lvl in range(n_levels_intermediate - 1):
            stencils.append(0.5 * stencils[-1])

    # Top level containing long-range tail (l = L), if included
    if include_toplevel:
        # TODO: sizes_toplevel must be larger than the top-level grid to avoid information loss
        #  during stencil computation but can be trimmed down to (2 * grid_shape - 1) later.
        #  -> Where and when should this happen?
        sizes_toplevel = tuple(
            s + len(omega) // 2 for s in grid_shape_toplevel
        )
        max_grid_level = n_levels_intermediate + 1
        distances_toplevel = get_distances(
            sizes_toplevel,
            2 ** (max_grid_level - 1) * spacings_or_gridcell_lvl_1,
        )
        kernel_values_at_gridpoints = k_toplevel(distances_toplevel)
        stencils.append(
            _compute_kernel_stencil_toplevel(
                kernel_values_at_gridpoints, omega
            )
        )

    return stencils


def suggest_max_grid_level_nonperiodic(
    side_lengths: ArrayLike,
    n_particles: int,
    level_one_spacings: ArrayLike,
    level_zero_cutoff: float,
    p: int,
):
    # 1. Find level at which spacing becomes larger than side length
    level_from_spacings = (
        onp.log2(side_lengths / level_one_spacings).astype(int) + 1
    )
    # TODO: min (spacings along NO direction greater than box size) or
    #  max (spacings along all but the longest side allowed to be greater than
    #  box size)?
    level_from_spacings = max(level_from_spacings)
    max_grid_level = level_from_spacings

    # 2. Find last level at which the cutoff is not larger than half the largest side length
    level_from_cutoff = onp.log2(side_lengths / level_zero_cutoff).astype(int)
    level_from_cutoff = max(level_from_cutoff)
    max_grid_level = min(max_grid_level, level_from_cutoff)

    # 3. Find (if achievable) the level where n_gridpoints <= sqrt(n_particles)
    shapes_all_levels, _ = set_up_grids_all_levels(
        side_lengths=side_lengths,
        level_one_spacings=level_one_spacings,
        max_grid_level=max_grid_level,
        p=p,
        pbc=(False,) * len(side_lengths),
    )
    n_gridpoints_all_levels = [
        None if s is None else onp.prod(s) for s in shapes_all_levels
    ]
    levels_with_fewer_points = (
        onp.where(n_gridpoints_all_levels[1:] <= onp.sqrt(n_particles))[0] + 1
    )
    if len(levels_with_fewer_points) > 0:
        max_grid_level = min(max_grid_level, min(levels_with_fewer_points))

    return int(max_grid_level)
