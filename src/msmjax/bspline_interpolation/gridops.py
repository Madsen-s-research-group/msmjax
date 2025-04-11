from functools import partial
from typing import Callable, Iterable, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt

from msmjax.bspline_interpolation.basis import create_bspline_basis_element
from msmjax.core.longrange import special_periodic_convolve


class BSplineInterpolationAxis(NamedTuple):
    length: float
    h: float
    p: int
    J_zeroplus: npt.ArrayLike
    periodic: bool
    n_domain: int
    n_total: int
    to_raw_indices: Callable
    from_raw_indices: Callable
    wrap_indices_if_periodic: Callable
    wrap_or_invalidate_indices: Callable
    evaluate_bspline_basis_for_one_particle: Callable
    evaluate_bspline_basis_multi: Callable
    evaluate_bspline_basis_gradient_multi: Callable


def set_up_grid_axis(
    length: float, h: float, p: int, J_zeroplus: npt.ArrayLike, periodic: bool
):
    if p % 2 != 0:
        raise ValueError("p must be even")

    if periodic:
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
        n_domain = int(onp.ceil(length / h))
        n_total = n_domain
    else:
        # TODO: Is this determination of the number of grid points numerically robust?
        #  OTOH, is it really a concern? (Is +1 actually necessary?)
        n_domain = int(onp.ceil(length / h)) + 1
        n_total = n_domain + p

    def to_raw_indices(indices):
        # TODO: name? (something like `zero_align_indices`?)
        if periodic:
            return indices
        else:
            return indices - p // 2

    def from_raw_indices(indices):
        # TODO: name?
        if periodic:
            return indices
        else:
            return indices + p // 2

    bspline_basis_element = create_bspline_basis_element(order=p - 1)

    def wrap_indices_if_periodic(indices):
        if periodic:
            return indices % n_total
        else:
            return indices

    def wrap_or_invalidate_indices(indices):
        if periodic:
            return indices % n_total
        else:
            intentionally_out_of_bounds_index = n_total
            # TODO: Checking only for negative indices would be enough, but perhapss less readable?
            is_in_bounds = jnp.logical_and(indices >= 0, indices < n_total)
            return jnp.where(
                is_in_bounds, indices, intentionally_out_of_bounds_index
            )

    def evaluate_bspline_basis_for_one_particle(
        x: float,
    ) -> Tuple[jax.Array, jax.Array]:
        x_over_h = x / h
        raw_reference_index = jnp.ceil(x_over_h).astype(int)
        # TODO: this could be onp.arange (to emphasize staticness, I don't think
        #  it makes a performance difference in this case).
        #  The same goes for other usags of jnp.arange as well.
        raw_indices = raw_reference_index + jnp.arange(-p // 2, p // 2)
        splinevals = jax.vmap(bspline_basis_element)(x_over_h - raw_indices)
        indices = wrap_indices_if_periodic(from_raw_indices(raw_indices))

        return splinevals, indices

    def evaluate_bspline_basis_multi(
        positions: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        return jax.vmap(evaluate_bspline_basis_for_one_particle)(positions)

    def evaluate_bspline_basis_gradient_multi(
        positions: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        # TODO: jacfwd vs. jacrev performance considerations?
        return jax.vmap(
            jax.jacfwd(evaluate_bspline_basis_for_one_particle, has_aux=True)
        )(positions)

    return BSplineInterpolationAxis(
        periodic=periodic,
        length=length,
        h=h,
        p=p,
        J_zeroplus=J_zeroplus,
        n_domain=n_domain,
        n_total=n_total,
        to_raw_indices=to_raw_indices,
        from_raw_indices=from_raw_indices,
        wrap_indices_if_periodic=wrap_indices_if_periodic,
        wrap_or_invalidate_indices=wrap_or_invalidate_indices,
        evaluate_bspline_basis_for_one_particle=evaluate_bspline_basis_for_one_particle,
        evaluate_bspline_basis_multi=evaluate_bspline_basis_multi,
        evaluate_bspline_basis_gradient_multi=evaluate_bspline_basis_gradient_multi,
    )


def arbitrary_dim_outer(*xi: jax.Array) -> jax.Array:
    """Compute the outer product of an arbitrary number of arrays"""
    # TODO: could this be made more efficient? (the way it is currently done
    #   technically involves redundant multiplications)
    return jnp.prod(jnp.array(jnp.meshgrid(*xi, indexing="ij")), axis=0)


def multi_inds_from_individual_axes_inds(*inds_individual_axes):
    # TODO: name of this function and its arguments?
    multi_inds = jnp.array(
        [
            arr.ravel()
            for arr in jnp.meshgrid(*inds_individual_axes, indexing="ij")
        ]
    ).T
    return multi_inds


class BSplineInterpolationGrid:
    def __init__(self, axes: List[BSplineInterpolationAxis]):
        self.axes = axes

        self.shape = tuple(g.n_total for g in axes)
        self.ndim = len(axes)
        self.size = int(onp.prod(self.shape))
        self.pbc = tuple(ax.periodic for ax in self.axes)

    def evaluate_bspline_basis_one_particle(self, position):
        spline_outputs_one_particle = [
            self.axes[idx_cartesian].evaluate_bspline_basis_for_one_particle(
                position[idx_cartesian]
            )
            for idx_cartesian in range(self.ndim)
        ]
        vals_individual_axes = [spl[0] for spl in spline_outputs_one_particle]
        inds_individual_axes = [spl[1] for spl in spline_outputs_one_particle]

        vals_flat = arbitrary_dim_outer(*vals_individual_axes).ravel()

        multi_inds = multi_inds_from_individual_axes_inds(
            *inds_individual_axes
        )
        # TODO: mode?
        # TODO: In fact, it would probably be helpful if this function
        #  returned out-of-bounds indices in case it receives coordinates
        #  that are outside the box, in order to facilitate catching errors
        #  (because particles are not supposed to move outside the pre-defined
        #  grid limits, and if they do something went wrong).
        #  Using ravel_multi_index, I don't think this behavior is possible,
        #  because it always clips or wraps indices to the valid range.
        #  `ravel_multi_inds_and_apply_bcs` from further below could be used?
        inds_flat = jax.vmap(
            lambda mi: jnp.ravel_multi_index(mi, dims=self.shape, mode="clip")
        )(multi_inds)

        return vals_flat, inds_flat

    def evaluate_bspline_basis_multiparticle(self, positions):
        return jax.vmap(self.evaluate_bspline_basis_one_particle)(positions)

    def evaluate_bspline_basis_gradient_multiparticle(self, positions):
        return jax.vmap(
            jax.jacfwd(self.evaluate_bspline_basis_one_particle, has_aux=True)
        )(positions)


def set_up_grids_all_levels(
    box_lengths: Iterable[float],
    level_one_spacings: Iterable[float],
    pbcs: Iterable[bool],
    n_levels: int,
    p: int,
    J_zeroplus: npt.ArrayLike,
):
    if n_levels < 1:
        raise ValueError("Need at least one grid level.")

    bspline_params = {"p": p, "J_zeroplus": J_zeroplus}

    # TODO: More flexible determination of grid spacing and number of levels
    #  - Non-periodic: option to leave out n_levels and determine from spacing
    #  - Periodic: if n_levels given but not spacing, determine spacing from n_levels
    #  - Periodic: if spacing given, but not n_levels, find n_levels for which
    #    the corresponding spacing most closely matches the one that was given
    #  - Mixed: ???
    #  - In general: check out how this is done in NAMD
    # TODO: commented part does not make sense?
    # actual_level_one_spacings = []
    # for length, spacing, periodic in zip(
    #     box_lengths, level_one_spacings, pbc
    # ):
    #     if periodic:
    #         actual_level_one_spacings.append(length / 2 ** (n_levels - 1))
    #     else:
    #         actual_level_one_spacings.append(spacing)

    grids_all_levels = [None]

    for lvl in range(1, n_levels + 1):
        axes = [
            set_up_grid_axis(
                length=length,
                h=2 ** (lvl - 1) * spacing,
                periodic=periodic,
                **bspline_params,
            )
            for length, spacing, periodic in zip(
                box_lengths, level_one_spacings, pbcs
            )
        ]
        grid = BSplineInterpolationGrid(axes)
        grids_all_levels.append(grid)

    return grids_all_levels


def make_ravel_multi_inds_and_apply_bcs(grid: BSplineInterpolationGrid):
    # TODO: This is not used anywhere except in (obsolete) custom interaction
    #  operator => remove?
    #  In fact, it might find a new use in the bspline basis evaluation
    #  (particles outside non-periodic boundaries should result in nans)
    # TODO: should this be a method of BSplineInterpolationGrid? -> probably not
    is_not_periodic = ~jnp.array([ga.periodic for ga in grid.axes])
    intentionally_out_of_bounds_index = grid.size

    def ravel_multi_inds_and_apply_bcs(multi_indices: jax.Array) -> jax.Array:
        # This handles periodic axes on its own by using the "wrap" keyword
        # TODO: Does the vmapping without transpose argument give the same
        #  as argument transpose with no vmap?
        # flat_inds = jax.vmap(
        #     lambda multi_index: jnp.ravel_multi_index(
        #         multi_index, dims=grid.shape, mode="wrap"
        #     )
        # )(multi_indices)
        flat_inds = jnp.ravel_multi_index(
            multi_indices.T, dims=grid.shape, mode="wrap"
        )
        # Explicitly handle non-periodic axes
        is_out_of_bounds = jnp.logical_or(
            multi_indices < 0, multi_indices >= jnp.array(grid.shape)
        )
        is_out_of_bounds = (is_out_of_bounds & is_not_periodic).any(axis=-1)
        flat_inds = jnp.where(
            is_out_of_bounds.ravel(),
            intentionally_out_of_bounds_index,
            flat_inds,
        )

        return flat_inds

    return ravel_multi_inds_and_apply_bcs


def create_restriction_operator_1d(
    axis_source_fine: BSplineInterpolationAxis,
    axis_target_coarse: BSplineInterpolationAxis,
) -> Callable:
    """Create function that performs the restriction operation"""
    # TODO: check if both grids have same J and p?
    # TODO: check if shape of J is compatible with p?
    p = axis_source_fine.p
    J_zeroplus = axis_source_fine.J_zeroplus
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    def get_neighbor_inds_on_sourcegrid(idx_targetgrid: int) -> jax.Array:
        """Get a target-grid index's neighbor indices on source grid."""
        raw_idx_targetgrid = axis_target_coarse.to_raw_indices(idx_targetgrid)
        raw_neighbor_inds_sourcegrid = 2 * raw_idx_targetgrid + jnp.arange(
            -p // 2, p // 2 + 1
        )
        neighbor_inds_sourcegrid = axis_source_fine.from_raw_indices(
            raw_neighbor_inds_sourcegrid
        )
        return neighbor_inds_sourcegrid

    inds_targetgrid = jnp.arange(axis_target_coarse.n_total)

    def restrict_1d(in_array_fine: jax.Array) -> jax.Array:
        """Restrict array defined on grid to the next-coarser (higher) grid"""
        neighbor_inds_sourcegrid = jax.vmap(get_neighbor_inds_on_sourcegrid)(
            inds_targetgrid
        )
        neighbor_inds_sourcegrid = axis_source_fine.wrap_or_invalidate_indices(
            neighbor_inds_sourcegrid
        )
        # TODO: replace `get` with a `jnp.where` construct? (more efficient?)
        #  (see create_interaction_operator_custom for an example)
        neighbor_values_sourcegrid = in_array_fine.at[
            neighbor_inds_sourcegrid
        ].get(mode="fill", fill_value=0.0)

        out_array_coarse = jnp.zeros(axis_target_coarse.n_total)
        # TODO: use `set` instead of `add`? Or get rid of the `at` altogether!
        out_array_coarse = out_array_coarse.at[inds_targetgrid].add(
            (neighbor_values_sourcegrid * J).sum(axis=1)
        )

        return out_array_coarse

    return restrict_1d


def create_restriction_operator(
    grid_source_fine: BSplineInterpolationGrid,
    grid_target_coarse: BSplineInterpolationGrid,
) -> Callable:
    restriction_funcs_1d_individual_axes = []
    for axis_source, axis_target in zip(
        grid_source_fine.axes, grid_target_coarse.axes
    ):
        restriction_funcs_1d_individual_axes.append(
            create_restriction_operator_1d(
                axis_source_fine=axis_source,
                axis_target_coarse=axis_target,
            )
        )

    def restrict(in_array_fine):
        out_array_coarse = in_array_fine

        for idx_cartesian, rf_1d in enumerate(
            restriction_funcs_1d_individual_axes
        ):
            out_array_coarse = jnp.apply_along_axis(
                func1d=rf_1d,
                axis=idx_cartesian,
                arr=out_array_coarse,
            )

        return out_array_coarse

    return restrict


def create_prolongation_operator_1d(
    axis_source_coarse: BSplineInterpolationAxis,
    axis_target_fine: BSplineInterpolationAxis,
):
    # TODO: check if both grids have same J and p?
    # TODO: check if shape of J is compatible with p?
    p = axis_source_coarse.p
    J_zeroplus = jnp.asarray(axis_source_coarse.J_zeroplus)

    start_even = int(onp.ceil(onp.round(-p / 4, decimals=1)))
    end_even = int(onp.floor(onp.round(p / 4, decimals=1)))
    start_odd = int(onp.ceil(onp.round(0.5 - p / 4, decimals=1)))
    end_odd = int(onp.floor(onp.round(0.5 + p / 4, decimals=1)))
    neighbor_distances_even_target_idx = jnp.arange(start_even, end_even + 1)
    neighbor_distances_odd_target_idx = jnp.arange(start_odd, end_odd + 1)

    inds_into_J_even = -2 * neighbor_distances_even_target_idx
    inds_into_J_odd = 1 - 2 * neighbor_distances_odd_target_idx

    def get_neighbor_inds_on_sourcegrid_even(idx_target_even: int):
        """Get an even target-grid index's neighbor indices on source grid"""
        raw_idx_target = axis_target_fine.to_raw_indices(idx_target_even)
        raw_neighbor_inds_source = (
            raw_idx_target // 2 + neighbor_distances_even_target_idx
        )
        neighbor_inds_source = axis_source_coarse.from_raw_indices(
            raw_neighbor_inds_source
        )
        return axis_source_coarse.wrap_indices_if_periodic(
            neighbor_inds_source
        )

    def get_neighbor_inds_on_sourcegrid_odd(idx_target_odd: int):
        """Get an odd target-grid index's neighbor indices on source grid"""
        raw_idx_target = axis_target_fine.to_raw_indices(idx_target_odd)
        raw_neighbor_inds_source = (
            raw_idx_target // 2 + neighbor_distances_odd_target_idx
        )
        neighbor_inds_source = axis_source_coarse.from_raw_indices(
            raw_neighbor_inds_source
        )
        return axis_source_coarse.wrap_indices_if_periodic(
            neighbor_inds_source
        )

    if axis_target_fine.periodic:
        slice_even = slice(0, None, 2)
        slice_odd = slice(1, None, 2)
    else:
        slice_even = slice((p // 2) % 2, None, 2)
        slice_odd = slice(1 - (p // 2) % 2, None, 2)

    inds_targetgrid = jnp.arange(axis_target_fine.n_total)

    def prolongate_1d(in_array_coarse: jax.Array) -> jax.Array:
        """Prolongate array defined on grid to the next-finer (lower) grid"""
        inds_source_even = jax.vmap(get_neighbor_inds_on_sourcegrid_even)(
            inds_targetgrid[slice_even]
        )
        inds_source_odd = jax.vmap(get_neighbor_inds_on_sourcegrid_odd)(
            inds_targetgrid[slice_odd]
        )
        out_array_fine = jnp.zeros(axis_target_fine.n_total)
        # TODO: use `set` instead of `add`? Or get rid of the `at` altogether!
        out_array_fine = out_array_fine.at[inds_targetgrid[slice_even]].add(
            (
                in_array_coarse[inds_source_even]
                * J_zeroplus[jnp.abs(inds_into_J_even)]
            ).sum(axis=1)
        )
        # TODO: use `set` instead of `add`? Or get rid of the `at` altogether!
        out_array_fine = out_array_fine.at[inds_targetgrid[slice_odd]].add(
            (
                in_array_coarse[inds_source_odd]
                * J_zeroplus[jnp.abs(inds_into_J_odd)]
            ).sum(axis=1)
        )

        return out_array_fine

    return prolongate_1d


def create_prolongation_operator(
    grid_source_coarse: BSplineInterpolationGrid,
    grid_target_fine: BSplineInterpolationGrid,
) -> Callable:
    prolongation_funcs_1d_individual_axes = []
    for axis_source, axis_target in zip(
        grid_source_coarse.axes, grid_target_fine.axes
    ):
        prolongation_funcs_1d_individual_axes.append(
            create_prolongation_operator_1d(
                axis_source_coarse=axis_source,
                axis_target_fine=axis_target,
            )
        )

    def prolongate(in_array_coarse):
        out_array_fine = in_array_coarse

        for idx_cartesian, pf_1d in enumerate(
            prolongation_funcs_1d_individual_axes
        ):
            out_array_fine = jnp.apply_along_axis(
                func1d=pf_1d,
                axis=idx_cartesian,
                arr=out_array_fine,
            )

        return out_array_fine

    return prolongate


def create_all_grid_to_grid_ops(grids, convolution_methods=None):
    """Create all necessary functions that map from grids to grids

    Args:
        convolution_methods:
    """
    n_levels = len(grids) - 1

    if convolution_methods is None:
        # TODO: Is this the best place to specify the convolution method? Do
        #  in the default parameters of a higher-level function instead?
        # TODO: Do the strings really need to contain "scipy", now that my
        #  "custom" convolution function has been removed? Aren't "direct"
        #  and "fft" enough?
        convolution_methods = [None] + ["scipy-fft"] * n_levels
    if isinstance(convolution_methods, str):
        convolution_methods = [None] + [convolution_methods] * n_levels

    restriction_fns = [None] * (n_levels + 1)
    for lvl in range(2, n_levels + 1):
        restrict = create_restriction_operator(
            grid_source_fine=grids[lvl - 1], grid_target_coarse=grids[lvl]
        )
        restriction_fns[lvl] = restrict

    prolongation_fns = [None] * (n_levels + 1)
    for lvl in range(1, n_levels):
        prolongate = create_prolongation_operator(
            grid_source_coarse=grids[lvl + 1], grid_target_fine=grids[lvl]
        )
        prolongation_fns[lvl] = prolongate

    # TODO: There might be more efficient ways to compute the convolution on
    #  the highest level for non-periodic cases (where the stencil is always
    #  larger than the grid)
    interaction_fns = [None] * (n_levels + 1)
    for lvl in range(1, n_levels + 1):
        conv_meth = convolution_methods[lvl]
        # TODO: test that all these convolution methods actually give the
        #  same result
        if conv_meth == "scipy-direct":
            interact = partial(
                special_periodic_convolve,
                pbc=grids[lvl].pbc,
                method="direct",
            )
        elif conv_meth == "scipy-fft":
            interact = partial(
                special_periodic_convolve,
                pbc=grids[lvl].pbc,
                method="fft",
            )
        else:
            raise ValueError(
                f"`{conv_meth}` is not a valid convolution method"
            )
        interaction_fns[lvl] = interact

    return restriction_fns, prolongation_fns, interaction_fns
