"""Generic code for long-range part (that is evaluated using grids).

References:
    [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
    R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
    Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
    144 (11), 114112. https://doi.org/10.1063/1.4943868.

    [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
    Forces for the Simulation of Biomolecules (PhD thesis), University
    of Illinois at Urbana-Champaign, 2006.
"""

from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from msmjax.bspline.basis import create_bspline_basis_element
from msmjax.bspline.coefficients import compute_j_zeroplus
from msmjax.core.longrange import special_periodic_convolve_scipy
from msmjax.utils.general import ConvMeth


def _find_n_gridpoints_1d(
    length: float, h: float, p: int, is_periodic: bool
) -> int:
    """Find number of grid points along one dimension.

    Args:
        length: Length of the domain where particles can be located.
        h: Grid spacing.
        p: Interpolation order.
        is_periodic: Whether there is periodicity along this dimension.

    Returns:
        Number of grid points.
    """
    if p % 2 != 0:
        raise ValueError("p must be even")

    if is_periodic:
        # This checks that h = length * 2**(-n) or h = (length / 3) * 2**(-n),
        # with an integer exponent n.
        exponent_one = onp.log2(length / h)
        exponent_three = onp.log2(length / (3 * h))
        if not (
            onp.isclose(exponent_one, onp.round(exponent_one, 0))
            or onp.isclose(exponent_three, onp.round(exponent_three, 0))
        ):
            raise ValueError(
                "Along any periodic axis, the grid spacing must be a power "
                "of two times either the box length or the box length "
                "divided by three."
            )
        return int(onp.ceil(length / h))
    else:
        # +1 to be safe
        return int(onp.ceil(length / h)) + 1 + p


def set_up_grids_all_levels(
    side_lengths: Sequence[float],
    level_one_spacings: Sequence[float],
    pbc: Sequence[bool],
    max_grid_level: int,
    p: int,
) -> tuple[list[None | tuple[int, ...]], list[None | onp.ndarray]]:
    """Determine the shapes and spacings of the grids at all levels.

    Args:
        side_lengths: Side lengths of a cuboid region where particles may be
            located.
        level_one_spacings: Level-one grid spacings, one for each direction.
        pbc: One boolean per direction signaling periodicity.
        max_grid_level: The highest grid level.
        p: Interpolation order.

    Returns:
        Tuple containing:
            - A list of tuples indicating grid shapes, with the tuple at
              index l of the list corresponding to grid level l.
            - A list of arrays indicating grid spacings, with the tuple at
              index l of the list corresponding to grid level l.
    """
    if max_grid_level < 1:
        raise ValueError("Need at least one grid level.")

    level_one_spacings = onp.asarray(level_one_spacings)

    shapes_all_levels = [None]
    spacings_all_levels = [None]

    for lvl in range(1, max_grid_level + 1):
        spacings = 2 ** (lvl - 1) * level_one_spacings
        shape = tuple(
            _find_n_gridpoints_1d(length, h, p, is_periodic)
            for length, h, is_periodic in zip(side_lengths, spacings, pbc)
        )
        shapes_all_levels.append(shape)
        spacings_all_levels.append(spacings)

    return shapes_all_levels, spacings_all_levels


def _arbitrary_dim_outer(*xi: Array) -> Array:
    """Compute the outer product of an arbitrary number of arrays"""
    return jnp.prod(jnp.array(jnp.meshgrid(*xi, indexing="ij")), axis=0)


def _multiindex_outer(*inds_individual_axes: Array) -> tuple[Array, ...]:
    """Construct a mesh grid and return it processed into a multiindex."""
    multi_inds = tuple(
        arr.ravel()
        for arr in jnp.meshgrid(*inds_individual_axes, indexing="ij")
    )
    return multi_inds


def _ravel_multi_index_with_invalidation(
    multi_index: tuple[Array, ...], dims: Sequence[int], pbc: Sequence[bool]
):
    """Like jax.numpy.ravel_multi_index() with different out-of-bounds handling

    The out-of-bounds handling of this function depends on the boundary
    conditions, specified via ``pbc``.
    Along periodic directions, it is identical to calling
    ``jax.numpy.ravel_multi_index()`` with ``mode="wrap"``.
    Along nonperiodic directions, any out-of-bounds values in the input
    mult-index are become out-of-bounds values also in the output flat index.

    Args:
        multi_index: Like in jax.numpy.ravel_multi_index().
        dims: Like in jax.numpy.ravel_multi_index().
        pbc: One boolean per direction signaling periodicity.

    Returns:
        Flat indices.
    """
    pbc = onp.asarray(pbc)
    all_periodic = pbc.all()
    any_periodic = pbc.any()
    intentionally_out_of_bounds_index = onp.prod(dims)

    flat_inds_wrapped = jnp.ravel_multi_index(multi_index, dims, mode="wrap")
    if all_periodic:
        return flat_inds_wrapped
    else:
        multi_index = jnp.asarray(multi_index)
        in_bounds = jnp.logical_and(
            multi_index >= 0, multi_index < jnp.array(dims)[:, jnp.newaxis]
        )
        if any_periodic:
            in_bounds = jnp.logical_or(in_bounds, pbc[:, jnp.newaxis])
        return jnp.where(
            in_bounds.all(axis=0),
            flat_inds_wrapped,
            intentionally_out_of_bounds_index,
        )


def make_basis_evaluation_fn(
    grid_shape: tuple[int, ...], p: int, pbc: Sequence[bool]
) -> Callable[[Array, Array], tuple[Array, Array]]:
    """Make fn that evaluates all contributing basis fns for a single particle.

    Args:
        grid_shape: Tuple of integers representing shape of the grid on which
            the basis functions are defined.
        p: Interpolation order.
        pbc: One boolean per direction signaling periodicity.

    Returns:
        A function that evaluates the grid-centered basis functions for a
        single particle. It has two inputs and two outputs:

        - The inputs are two 1-d arrays, both of shape ``(n_dim, )``,
          representing the coordinates of a single particle, and the grid
          spacings along each direction.

        - The outputs are two 1-d arrays, representing the values of all
          basis functions that contain the particle in their support, and the
          flat indices of the corresponding grid points. They are always
          flat arrays, regardless of the spatial dimension of the system.
    """
    pbc = onp.asarray(pbc)

    def zero_align_idx(positional_idx):
        if pbc.all():
            return positional_idx
        elif (~pbc).all():
            return positional_idx - p // 2
        else:
            return jnp.where(
                pbc[:, jnp.newaxis],
                positional_idx,
                positional_idx - p // 2,
            )

    def to_positional_idx(zero_aligned_idx):
        if pbc.all():
            return zero_aligned_idx
        elif (~pbc).all():
            return zero_aligned_idx + p // 2
        else:
            return jnp.where(
                pbc[:, jnp.newaxis],
                zero_aligned_idx,
                zero_aligned_idx + p // 2,
            )

    bspline_basis_element = create_bspline_basis_element(order=p - 1)

    def eval_basis(coords: Array, spacings: Array) -> tuple[Array, Array]:
        r_over_h = coords / spacings
        raw_reference_inds = jnp.ceil(r_over_h).astype(int)
        raw_inds = raw_reference_inds[:, jnp.newaxis] + jnp.arange(
            -p // 2, p // 2
        )
        splinevals_per_axis = jax.vmap(jax.vmap(bspline_basis_element))(
            (r_over_h - raw_inds.T).T
        )
        inds_per_axis = to_positional_idx(raw_inds)

        splinevals = _arbitrary_dim_outer(*splinevals_per_axis).ravel()

        multiinds = _multiindex_outer(*inds_per_axis)
        flat_inds = _ravel_multi_index_with_invalidation(
            multiinds, dims=grid_shape, pbc=pbc
        )

        return splinevals, flat_inds

    return eval_basis


def _make_restrict_1d(
    n_points_in: int, n_points_out: int, p: int, is_periodic: bool
) -> Callable[[Array], Array]:
    """Make a function that performs restriction along one dimension."""
    J_zeroplus = compute_j_zeroplus(p)
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    def zero_align_idx(positional_idx):
        if is_periodic:
            return positional_idx
        else:
            return positional_idx - p // 2

    def to_positional_idx(zero_aligned_idx):
        if is_periodic:
            return zero_aligned_idx
        else:
            return zero_aligned_idx + p // 2

    def restrict_1d(in_array_fine: jax.Array) -> jax.Array:
        i = jnp.arange(n_points_out)
        i_aligned = zero_align_idx(i)
        j_aligned = (2 * i_aligned)[:, jnp.newaxis] + jnp.arange(
            -p // 2, p // 2 + 1
        )
        j = to_positional_idx(j_aligned)

        if is_periodic:
            j = j % n_points_in
            selected_source_values = in_array_fine[j]
        else:
            selected_source_values = jnp.where(
                jnp.logical_and(j >= 0, j < n_points_in),
                in_array_fine[j],
                0.0,
            )

        return (selected_source_values * J).sum(axis=1)

    return restrict_1d


def make_restriction_operator(
    grid_shape_in: Sequence[int],
    grid_shape_out: Sequence[int],
    p: int,
    pbc: Sequence[bool],
) -> Callable[[Array], Array]:
    """Make a function that performs restriction.

    Args:
        grid_shape_in: Tuple of integers indicating the shape of the input
            (lower-level, finer) grid.
        grid_shape_out: Tuple of integers indicating the shape of the target
            (higher-level, coarser) grid.
        p: Interpolation order.
        pbc: One boolean per direction signaling periodicity.

    Returns:
        A function that takes in an array of shape ``grid_shape_in`` and
        restricts it to a coarser grid, returning an array of shape
        ``grid_shape_out``.
    """
    restriction_fns_1d_per_axis = [
        _make_restrict_1d(n_points_in, n_points_out, p, is_periodic)
        for n_points_in, n_points_out, is_periodic in zip(
            grid_shape_in, grid_shape_out, pbc
        )
    ]

    def restrict(in_array_fine):
        out_array_coarse = in_array_fine
        for axis, fn_1d in enumerate(restriction_fns_1d_per_axis):
            out_array_coarse = jnp.apply_along_axis(
                func1d=fn_1d,
                axis=axis,
                arr=out_array_coarse,
            )
        return out_array_coarse

    return restrict


def _make_prolongate_1d(
    n_points_in: int, n_points_out: int, p: int, is_periodic: bool
) -> Callable[[Array], Array]:
    """Make a function that performs prolongation along one dimension."""
    J_zeroplus = jnp.array(compute_j_zeroplus(p))
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    def zero_align_idx(positional_idx):
        if is_periodic:
            return positional_idx
        else:
            return positional_idx - p // 2

    def to_positional_idx(zero_aligned_idx):
        if is_periodic:
            return zero_aligned_idx
        else:
            return zero_aligned_idx + p // 2

    start_even = int(onp.ceil(onp.round(-p / 4, decimals=1)))
    end_even = int(onp.floor(onp.round(p / 4, decimals=1)))
    start_odd = int(onp.ceil(onp.round(0.5 - p / 4, decimals=1)))
    end_odd = int(onp.floor(onp.round(0.5 + p / 4, decimals=1)))
    dists_to_neighbors_even = jnp.arange(start_even, end_even + 1)
    dists_to_neighbors_odd = jnp.arange(start_odd, end_odd + 1)

    inds_into_J_even = -2 * dists_to_neighbors_even
    inds_into_J_odd = 1 - 2 * dists_to_neighbors_odd

    if is_periodic:
        slice_even = slice(0, None, 2)
        slice_odd = slice(1, None, 2)
    else:
        slice_even = slice((p // 2) % 2, None, 2)
        slice_odd = slice(1 - (p // 2) % 2, None, 2)

    def prolongate_1d(in_array_coarse: jax.Array) -> jax.Array:
        i = jnp.arange(n_points_out)

        i_even = i[slice_even]
        i_even_aligned = zero_align_idx(i_even)
        j_even_aligned = (i_even_aligned // 2)[
            :, jnp.newaxis
        ] + dists_to_neighbors_even
        j_even = to_positional_idx(j_even_aligned)
        if is_periodic:
            j_even = j_even % n_points_in

        i_odd = i[slice_odd]
        i_odd_aligned = zero_align_idx(i_odd)
        j_odd_aligned = (i_odd_aligned // 2)[
            :, jnp.newaxis
        ] + dists_to_neighbors_odd
        j_odd = to_positional_idx(j_odd_aligned)
        if is_periodic:
            j_odd = j_odd % n_points_in

        result = jnp.zeros(n_points_out)
        result = result.at[i_even].add(
            (
                in_array_coarse[j_even] * J_zeroplus[jnp.abs(inds_into_J_even)]
            ).sum(axis=1)
        )
        result = result.at[i_odd].add(
            (
                in_array_coarse[j_odd] * J_zeroplus[jnp.abs(inds_into_J_odd)]
            ).sum(axis=1)
        )

        return result

    return prolongate_1d


def make_prolongation_operator(
    grid_shape_in: Sequence[int],
    grid_shape_out: Sequence[int],
    p: int,
    pbc: Sequence[bool],
) -> Callable[[Array], Array]:
    """Make a function that performs prolongation.

    Args:
        grid_shape_in: Tuple of integers indicating the shape of the input
            (higher-level, coarser) grid.
        grid_shape_out: Tuple of integers indicating the shape of the target
            (lower-level, finer) grid.
        p: Interpolation order.
        pbc: One boolean per direction signaling periodicity.

    Returns:
        A function that takes in an array of shape ``grid_shape_in`` and
        prolongs it to a finer grid, returning an array of shape
        ``grid_shape_out``.
    """
    prolongation_fns_1d_per_axis = [
        _make_prolongate_1d(n_points_in, n_points_out, p, is_periodic)
        for n_points_in, n_points_out, is_periodic in zip(
            grid_shape_in, grid_shape_out, pbc
        )
    ]

    def prolongate(in_array_coarse):
        out_array_fine = in_array_coarse
        for axis, fn_1d in enumerate(prolongation_fns_1d_per_axis):
            out_array_fine = jnp.apply_along_axis(
                func1d=fn_1d,
                axis=axis,
                arr=out_array_fine,
            )
        return out_array_fine

    return prolongate


def create_all_grid_to_grid_ops(
    grid_shapes: Sequence[tuple[int, ...]],
    p: int,
    pbc: Sequence[bool],
    convolution_methods: Sequence[None | ConvMeth],
):
    """Create all necessary functions that map from grids to grids

    Args:
        grid_shapes: Sequence of tuples of integers indicating the shape of
            the grids at all levels.
        p: Interpolation order.
        pbc: One boolean per direction signaling periodicity.
        convolution_methods: Algorithm to use for convolution in the grid
            potential calculation. One value per grid level (i.e., different
            algorithms can be used at different levels).

    Returns:
        Three-element tuple containing:
            - Sequence of functions for restrictions between all pairs of
              grids at adjacent levels.
            - Sequence of functions for prolongations between all pairs of
              grids at adjacent levels.
            - Sequence of functions for potential calculation via convolution
              with kernel stencils at all levels.
    """
    max_level_grids = len(grid_shapes) - 1

    restriction_fns = [None] * (max_level_grids + 1)
    for lvl in range(2, max_level_grids + 1):
        restrict = make_restriction_operator(
            grid_shape_in=grid_shapes[lvl - 1],
            grid_shape_out=grid_shapes[lvl],
            p=p,
            pbc=pbc,
        )
        restriction_fns[lvl] = restrict

    prolongation_fns = [None] * (max_level_grids + 1)
    for lvl in range(1, max_level_grids):
        prolongate = make_prolongation_operator(
            grid_shape_in=grid_shapes[lvl + 1],
            grid_shape_out=grid_shapes[lvl],
            p=p,
            pbc=pbc,
        )
        prolongation_fns[lvl] = prolongate

    convolution_fns = [None] * (max_level_grids + 1)
    for lvl in range(1, max_level_grids + 1):
        conv_meth = convolution_methods[lvl]
        if conv_meth == "scipy-direct":
            interact = partial(
                special_periodic_convolve_scipy,
                pbc=pbc,
                method="direct",
            )
        elif conv_meth == "scipy-fft":
            interact = partial(
                special_periodic_convolve_scipy,
                pbc=pbc,
                method="fft",
            )
        else:
            raise ValueError(
                f"`{conv_meth}` is not a valid convolution method"
            )
        convolution_fns[lvl] = interact

    return restriction_fns, prolongation_fns, convolution_fns


def suggest_max_grid_level_nonperiodic(
    side_lengths: ArrayLike,
    n_particles: int,
    level_one_spacings: ArrayLike,
    level_zero_cutoff: float,
    p: int,
) -> int:
    """Suggest an efficient value for highest grid level in nonperiodic case.

    The criterion used, as per Ref. [1], is that the number of grid points
    at the highest level should be less than or equal to the sqare root of the
    number of particles. However, this may be unachievable for lower particle
    counts, since the support of the basis functions dictates that the grids
    extend beyond the boundary of the cell, so there is a minimum number of
    grid points that cannot be reduced further even by going to arbitrarily
    high grid levels.
    Therefore, other criteria are implemented as a failsafe as well.

    Args:
        side_lengths: Side lengths of a cuboid region where particles may be
            located.
        n_particles: Number of particles.
        level_one_spacings: Level-one grid spacings, one value per direction.
        level_zero_cutoff: Level-zero cutoff.
        p: Interpolation order.

    Returns:
        Suggested value for highest grid level.
    """
    error_message = (
        "Automatic determination of the number of grid levels for "
        "non-periodic system resulted in an invalid value less than 1. "
        "Are you using a a too large (compared to the simulation cell) "
        "level-one grid spacing or level-zero cutoff?"
    )

    # 1. Find level at which spacing becomes larger than side length.
    #    This corresponds to the point where the number of grid points cannot
    #    be reduced any further by adding another level, and serves as an upper
    #    limit on the number of grid levels.
    upper_limit_per_direction = (
        onp.log2(side_lengths / level_one_spacings).astype(int) + 1
    )
    max_grid_level = max(upper_limit_per_direction)

    # 2. Find the highest level at which the cutoff is not larger than half
    #    the longest side of the cell.
    level_from_cutoff = onp.log2(side_lengths / level_zero_cutoff).astype(int)
    level_from_cutoff = max(level_from_cutoff)
    max_grid_level = min(max_grid_level, level_from_cutoff)

    if max_grid_level < 1:
        raise ValueError(error_message)

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

    if max_grid_level < 1:
        raise ValueError(error_message)

    return int(max_grid_level)


def find_spacings_and_max_level_periodic(
    side_lengths, target_level_one_spacings
):
    """Find grid spacings and number of grid levels compatible with periodicity

    The spacing and number of levels are not independent and are
    adjusted/determined automatically because:

    - In a periodic system, the grid spacings need to divide the side
      lengths without remainder.
    - In our implementation of the MSM for periodic boundary conditions,
      based on Ref. [2], the number of grids keeps being increased until
      there is only one point at the highest level, which is then not
      actually visited (it would not contribute due to charge neutrality).
      Since this condition can lead to non-ideal spacings, far from the
      ideal ones, being enforced, we allow the grid points along each
      direction to be not just powers of two, but also three times a power
      of two.

    Args:
        side_lengths: Array of side lengths, one per direction.
        target_level_one_spacings: Array of target grid spacings at level one.

    Returns:
        Tuple containing:
            - Array of level-one spacings that have been adjusted to be
              compatible with periodic boundary conditions.
            - The highest level in the kernel splitting. This is one level
              higher than the highest grid level at which calculations are
              actually performed, because the top level (with either one or
              three grid points in each direction) is omitted.
    """
    side_lengths = onp.array(side_lengths)
    target_level_one_spacings = onp.array(target_level_one_spacings)

    raw_ells_one_based = onp.log2(side_lengths / target_level_one_spacings) + 1
    candidate_ells_one_based = onp.column_stack(
        [onp.floor(raw_ells_one_based), onp.ceil(raw_ells_one_based)]
    )
    candidate_spacings_one_based = side_lengths[:, onp.newaxis] / 2 ** (
        candidate_ells_one_based - 1
    )
    raw_ells_three_based = (
        onp.log2(side_lengths / (3 * target_level_one_spacings)) + 2
    )
    candidate_ells_three_based = onp.column_stack(
        [onp.floor(raw_ells_three_based), onp.ceil(raw_ells_three_based)]
    )
    candidate_spacings_three_based = side_lengths[:, onp.newaxis] / (
        3 * 2 ** (candidate_ells_three_based - 2)
    )

    candidate_spacings = onp.concatenate(
        [candidate_spacings_one_based, candidate_spacings_three_based], axis=1
    )
    candidate_ells = onp.concatenate(
        [candidate_ells_one_based, candidate_ells_three_based], axis=1
    ).astype(int)
    deviations = onp.abs(
        candidate_spacings - target_level_one_spacings[:, onp.newaxis]
    )
    inds_best_match = onp.argmin(deviations, axis=1)
    ells = candidate_ells[onp.arange(candidate_ells.shape[0]), inds_best_match]
    adjusted_spacings = candidate_spacings[
        onp.arange(candidate_spacings.shape[0]), inds_best_match
    ]
    max_splitting_level = int(max(ells))

    # Remember that in periodic cases the highest grid level that is actually
    # evaluated is one less than the highest splitting level we just determined
    if (max_splitting_level - 1) < 1:
        raise ValueError(
            "Automatic determination of the number of grid levels for "
            "periodic system resulted in an invalid value less than 1. "
            "Are you using a a too large (compared to the unit cell) "
            "level-one grid spacing?"
        )

    return adjusted_spacings, max_splitting_level
