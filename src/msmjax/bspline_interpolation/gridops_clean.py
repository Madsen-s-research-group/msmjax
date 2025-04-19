from functools import partial
from typing import Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from msmjax.bspline_interpolation.basis import create_bspline_basis_element
from msmjax.bspline_interpolation.coefficients import compute_J_zeroplus
from msmjax.core.longrange import special_periodic_convolve

# TODO: define somewhere central
ConvMeth = Literal["direct", "fft"]


def _find_n_gridpoints_1d(
    length: float, h: float, p: int, is_periodic: bool
) -> int:
    if p % 2 != 0:
        raise ValueError("p must be even")

    if is_periodic:
        if not (
            onp.isclose(length % h, 0.0)
            or onp.isclose(length % h, h)
            or onp.isclose(h % length, 0.0)
            or onp.isclose(h % length, length)
        ):
            raise ValueError(
                "Along any periodic axis, the grid spacing must either evenly "
                "divide the box length or be a multiple thereof."
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
    if max_grid_level < 1:
        raise ValueError("Need at least one grid level.")

    # TODO: Check same length of side_lengths, level_one_spacings, pbc?

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
    multi_inds = tuple(
        arr.ravel()
        for arr in jnp.meshgrid(*inds_individual_axes, indexing="ij")
    )
    return multi_inds


def _ravel_multi_index_with_invalidation(
    multi_index: tuple[Array, ...], dims: Sequence[int], pbc: Sequence[bool]
):
    # TODO: jit with static pbc?
    # TODO: function name?
    pbc = onp.asarray(pbc)  # TODO: onp/jnp?
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
            # TODO: Which axis? Shouldn't it be .all() instead of .any()?
            in_bounds.all(axis=0),
            flat_inds_wrapped,
            intentionally_out_of_bounds_index,
        )


def make_basis_evaluation_fn(
    grid_shape: tuple[int, ...], p: int, pbc: Sequence[bool]
) -> Callable[[Array, Array], tuple[Array, Array]]:
    pbc = onp.asarray(pbc)

    # TODO: External factory function that creates both `zero_align_idx` and
    #  `to_positional_idx` from `periodic` and `p`. Function names?

    def zero_align_idx(positional_idx):
        if pbc.all():
            return positional_idx
        elif (~pbc).all():
            return positional_idx - p // 2
        else:
            raise ValueError  # TODO: mixed BCs

    def to_positional_idx(zero_aligned_idx):
        if pbc.all():
            return zero_aligned_idx
        elif (~pbc).all():
            return zero_aligned_idx + p // 2
        else:
            raise ValueError  # TODO: mixed BCs

    bspline_basis_element = create_bspline_basis_element(order=p - 1)

    def eval_basis(coords: Array, spacings: Array) -> tuple[Array, Array]:
        # TODO: should spacing be argument to the closure or to the setup fn?
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


def _make_restrict_1d(n_points_in, n_points_out, p, is_periodic):
    # TODO: Check n_points_in >= n_points_out? Indicate 'fine' and 'coarse'
    #  by the variable names somehow?
    # TODO: Take n_points_in from the shape of the input array?

    # TODO: Variable names? Shouldn't be uppercase, and (lower-case) J is
    #  already in use as an index further down
    J_zeroplus = compute_J_zeroplus(p)
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    # TODO: External factory function that creates both `zero_align_idx` and
    #  `to_positional_idx` from `periodic` and `p`. Function names?

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
        # TODO: onp backend for these index arrays that could as well be static?
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

        # TODO: check that the output has the same length as axis_target_coarse?
        # TODO: Handle restricting from one point to one point (see branch `one_to_one_edge_cases`)!
        #  Also, does "min number of points -> min number of points" (in nonperiodic case) work correctly?

        return (selected_source_values * J).sum(axis=1)

    return restrict_1d


def make_restriction_operator(
    grid_shape_in: Sequence[int],
    grid_shape_out: Sequence[int],
    p: int,
    pbc: Sequence[bool],
) -> Callable[[Array], Array]:

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


def _make_prolongate_1d(n_points_in, n_points_out, p, is_periodic):
    # TODO: Check n_points_in <= n_points_out? Indicate 'fine' and 'coarse'
    #  by the variable names somehow?
    # TODO: Take n_points_in from the shape of the input array?

    # TODO: Variable names? Shouldn't be uppercase, and (lower-case) J is
    #  already in use as an index further down
    J_zeroplus = jnp.array(compute_J_zeroplus(p))
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    # TODO: External factory function that creates both `zero_align_idx` and
    #  `to_positional_idx` from `periodic` and `p`. Function names?

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

        # TODO: check that the output has the same length as axis_target_coarse?
        # TODO: Handle prolongating from one point to one point (see branch `one_to_one_edge_cases`)!
        #  Also, does "min number of points -> min number of points" (in nonperiodic case) work correctly?

        return result

    return prolongate_1d


def make_prolongation_operator(
    grid_shape_in: Sequence[int],
    grid_shape_out: Sequence[int],
    p: int,
    pbc: Sequence[bool],
) -> Callable[[Array], Array]:

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
    """Create all necessary functions that map from grids to grids"""
    max_level_grids = len(grid_shapes) - 1

    # TODO: check grid_shapes and convolution_methods same length
    # TODO: check pbc same length as the elements of grid_shapes

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

    # TODO: There might be more efficient ways to compute the convolution on
    #  the highest level for non-periodic cases (where the stencil is always
    #  larger than the grid)
    convolution_fns = [None] * (max_level_grids + 1)
    for lvl in range(1, max_level_grids + 1):
        conv_meth = convolution_methods[lvl]
        if conv_meth == "scipy-direct":
            interact = partial(
                special_periodic_convolve,
                pbc=pbc,
                method="direct",
            )
        elif conv_meth == "scipy-fft":
            interact = partial(
                special_periodic_convolve,
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
):
    # TODO: Put this function in this module or in some utils?

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


def find_spacings_and_max_level_periodic(
    side_lengths, target_level_one_spacings
):
    # TODO: Put this function in this module or in some utils?

    side_lengths = onp.array(side_lengths)
    target_level_one_spacings = onp.array(target_level_one_spacings)

    ells_raw_one_based = onp.log2(side_lengths / target_level_one_spacings) + 1
    candidate_ells_one_based = [
        onp.floor(ells_raw_one_based).astype(int),
        onp.ceil(ells_raw_one_based).astype(int),
    ]
    candidate_spacings_one_based = [
        side_lengths / 2 ** (onp.ceil(ells) - 1)
        for ells in candidate_ells_one_based
    ]

    ells_raw_three_based = (
        onp.log2(side_lengths / (3 * target_level_one_spacings)) + 2
    )
    candidate_ells_three_based = [
        onp.floor(ells_raw_three_based).astype(int),
        onp.ceil(ells_raw_three_based).astype(int),
    ]
    candidate_spacings_three_based = [
        side_lengths / (3 * 2 ** (ells - 2))
        for ells in candidate_ells_three_based
    ]

    candidate_spacings = onp.concatenate(
        [*candidate_spacings_one_based, *candidate_spacings_three_based]
    ).T
    candidate_ells = onp.concatenate(
        [*candidate_ells_one_based, *candidate_ells_three_based]
    ).T

    deviations = onp.abs(
        candidate_spacings - target_level_one_spacings[:, onp.newaxis]
    )
    inds_best_match = onp.argmin(deviations, axis=1)

    ells = candidate_ells[onp.arange(candidate_ells.shape[0]), inds_best_match]
    adjusted_spacings = candidate_spacings[
        onp.arange(candidate_spacings.shape[0]), inds_best_match
    ]

    return int(max(ells)), adjusted_spacings
