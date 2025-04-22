import numpy as onp
from jax.typing import ArrayLike

from msmjax.bspline.gridops_clean import set_up_grids_all_levels


def suggest_max_grid_level_nonperiodic(
    side_lengths: ArrayLike,
    n_particles: int,
    level_one_spacings: ArrayLike,
    level_zero_cutoff: float,
    p: int,
):
    # 1. Find level at which spacing becomes larger than side length.
    #    This corresponds to the point where the number of grid points cannot
    #    be reduced any further by adding another level, and serves as an upper
    #    limit on the number of grid levels.
    upper_limit_per_direction = (
        onp.log2(side_lengths / level_one_spacings).astype(int) + 1
    )
    max_grid_level = max(upper_limit_per_direction)

    # TODO: better explanation
    # 2. Find the highest level at which the cutoff is not larger than half
    #    the longest side of the cell.
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
