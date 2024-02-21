from functools import partial
from typing import Callable, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt

from msmjax.bspline_basis import create_bspline_basis_element


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
        n_domain = int(onp.ceil(length / h))
        n_total = n_domain
    else:
        # TODO: determination of number of grid points might not be numerically robust
        n_domain = int(onp.ceil(length / h)) + 1
        n_total = n_domain + p

    def to_raw_indices(indices):
        if periodic:
            return indices
        else:
            return indices - p // 2

    def from_raw_indices(indices):
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


def create_anterpolation_operator(grid: BSplineInterpolationGrid):
    """Create a function that anterpolates charge from particles to grid"""

    def anterpolate(positions: jax.Array, charges: jax.Array) -> jax.Array:
        """Anterpolate charge from particles to grid"""
        splinevals, indices = grid.evaluate_bspline_basis_multiparticle(
            positions
        )
        gridcharge = jnp.zeros(grid.size)
        gridcharge = gridcharge.at[indices].add(
            charges[:, jnp.newaxis] * splinevals
        )

        return gridcharge

    return anterpolate


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

    def get_neigbhor_inds_on_sourcegrid(idx_targetgrid: int) -> jax.Array:
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

    def restrict(in_array_fine: jax.Array) -> jax.Array:
        """Restrict array defined on grid to the next-coarser (higher) grid"""
        neighbor_inds_sourcegrid = jax.vmap(get_neigbhor_inds_on_sourcegrid)(
            inds_targetgrid
        )
        neighbor_inds_sourcegrid = axis_source_fine.wrap_or_invalidate_indices(
            neighbor_inds_sourcegrid
        )
        neighbor_values_sourcegrid = in_array_fine.at[
            neighbor_inds_sourcegrid
        ].get(mode="fill", fill_value=0.0)

        out_array_coarse = jnp.zeros(axis_target_coarse.n_total)
        out_array_coarse = out_array_coarse.at[inds_targetgrid].add(
            (neighbor_values_sourcegrid * J).sum(axis=1)
        )

        return out_array_coarse

    return restrict


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

        for idx_cartesian, restriction_func in enumerate(
            restriction_funcs_1d_individual_axes
        ):
            out_array_coarse = jnp.apply_along_axis(
                func1d=restriction_func,
                axis=idx_cartesian,
                arr=out_array_coarse,
            )

        return out_array_coarse

    return restrict


def create_prolongation_operator(
    grid_source_coarse: BSplineInterpolationAxis,
    grid_target_fine: BSplineInterpolationAxis,
):
    # TODO: check if both grids have same J and p?
    # TODO: check if shape of J is compatible with p?
    p = grid_source_coarse.p
    J_zeroplus = jnp.asarray(grid_source_coarse.J_zeroplus)

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
        raw_idx_target = grid_target_fine.to_raw_indices(idx_target_even)
        raw_neighbor_inds_source = (
            raw_idx_target // 2 + neighbor_distances_even_target_idx
        )
        neighbor_inds_source = grid_source_coarse.from_raw_indices(
            raw_neighbor_inds_source
        )
        return grid_source_coarse.wrap_indices_if_periodic(
            neighbor_inds_source
        )

    def get_neighbor_inds_on_sourcegrid_odd(idx_target_odd: int):
        """Get an odd target-grid index's neighbor indices on source grid"""
        raw_idx_target = grid_target_fine.to_raw_indices(idx_target_odd)
        raw_neighbor_inds_source = (
            raw_idx_target // 2 + neighbor_distances_odd_target_idx
        )
        neighbor_inds_source = grid_source_coarse.from_raw_indices(
            raw_neighbor_inds_source
        )
        return grid_source_coarse.wrap_indices_if_periodic(
            neighbor_inds_source
        )

    if grid_target_fine.periodic:
        slice_even = slice(0, None, 2)
        slice_odd = slice(1, None, 2)
    else:
        slice_even = slice((p // 2) % 2, None, 2)
        slice_odd = slice(1 - (p // 2) % 2, None, 2)

    inds_targetgrid = jnp.arange(grid_target_fine.n_total)

    def prolongate(in_array_coarse: jax.Array) -> jax.Array:
        """Prolongate array defined on grid to the next-finer (lower) grid"""
        inds_source_even = jax.vmap(get_neighbor_inds_on_sourcegrid_even)(
            inds_targetgrid[slice_even]
        )
        inds_source_odd = jax.vmap(get_neighbor_inds_on_sourcegrid_odd)(
            inds_targetgrid[slice_odd]
        )
        out_array_fine = jnp.zeros(grid_target_fine.n_total)
        out_array_fine = out_array_fine.at[inds_targetgrid[slice_even]].add(
            (
                in_array_coarse[inds_source_even]
                * J_zeroplus[jnp.abs(inds_into_J_even)]
            ).sum(axis=1)
        )
        out_array_fine = out_array_fine.at[inds_targetgrid[slice_odd]].add(
            (
                in_array_coarse[inds_source_odd]
                * J_zeroplus[jnp.abs(inds_into_J_odd)]
            ).sum(axis=1)
        )

        return out_array_fine

    return prolongate


def create_interaction_operator(
    grid: BSplineInterpolationAxis, kernel_stencil: npt.ArrayLike
):
    """Create a function that computes the interaction at one grid level"""
    kernel_stencil = jnp.asarray(kernel_stencil)
    interaction_range = len(kernel_stencil) // 2
    gridsize = grid.n_total

    def apply_interaction(in_array):
        """Convolve array defined on grid with interaction kernel stencil."""
        inds = jnp.arange(gridsize)
        neighbor_inds = inds[:, jnp.newaxis] + jnp.arange(
            -interaction_range, interaction_range + 1
        )
        neighbor_inds = grid.wrap_or_invalidate_indices(neighbor_inds)
        neighbor_values = in_array.at[neighbor_inds].get(
            mode="fill", fill_value=0.0
        )
        out_array = jnp.zeros(gridsize)
        out_array = out_array.at[inds].add(
            (neighbor_values * kernel_stencil).sum(axis=1)
        )

        return out_array

    return apply_interaction


def create_all_grid_to_grid_ops(grids, kernel_stencils):
    """Create all necessary functions that map from grids to grids"""
    max_gridlevel = len(grids) - 1

    restriction_funcs = [None] * (max_gridlevel + 1)
    for lvl in range(2, max_gridlevel + 1):
        restrict = create_restriction_operator(
            grid_source_fine=grids[lvl - 1], grid_target_coarse=grids[lvl]
        )
        restriction_funcs[lvl] = restrict

    prolongation_funcs = [None] * (max_gridlevel + 1)
    for lvl in range(1, max_gridlevel):
        prolongate = create_prolongation_operator(
            grid_source_coarse=grids[lvl + 1], grid_target_fine=grids[lvl]
        )
        prolongation_funcs[lvl] = prolongate

    interaction_funcs = [None] * (max_gridlevel + 1)
    for lvl in range(1, max_gridlevel + 1):
        interact = create_interaction_operator(
            grid=grids[lvl], kernel_stencil=kernel_stencils[lvl]
        )
        interaction_funcs[lvl] = interact

    return restriction_funcs, prolongation_funcs, interaction_funcs


def create_compute_gridpotential_level_one(grids, kernel_stencils) -> Callable:
    """Create closure for computing potential on lowest-level grid"""
    # TODO: check if all grids have same J and p?
    # TODO: check if shape of J is compatible with p?
    (
        restriction_funcs,
        prolongation_funcs,
        interaction_funcs,
    ) = create_all_grid_to_grid_ops(grids, kernel_stencils)

    def compute_gridpotential_level_one(
        gridcharge_level_one: jax.Array,
    ) -> jax.Array:
        """Compute level-one grid potential from level-one grid charge.

        Computes the quantity called e^{1+} in the reference article.

        This corresponds to going from the lowest grid level on the left side
        of ladder in Fig. 5 of the reference article all the way up to the
        highest grid level and back down to the lowest grid level on the right.

        Args:
            gridcharge_level_one: Array of grid charge at lowest grid level

        Returns:
            Accumulated potential at the lowest grid level
        """
        max_gridlevel = len(grids) - 1
        gridcharges_all_levels = {1: gridcharge_level_one}

        # Go up ladder
        for lvl in range(2, max_gridlevel + 1):
            restrict = restriction_funcs[lvl]
            gridcharge_fine = gridcharges_all_levels[lvl - 1]
            gridcharge_coarse = restrict(gridcharge_fine)
            gridcharges_all_levels[lvl] = gridcharge_coarse

        # Apply top-level interaction
        gridcharge_toplevel = gridcharges_all_levels[max_gridlevel]
        interact_toplevel = interaction_funcs[max_gridlevel]
        gridpotential = interact_toplevel(gridcharge_toplevel)

        # Go down ladder
        for lvl in range(max_gridlevel - 1, 0, -1):
            gridpotential = interaction_funcs[lvl](
                gridcharges_all_levels[lvl]
            ) + prolongation_funcs[lvl](gridpotential)

        return gridpotential

    return compute_gridpotential_level_one


def create_compute_U_oneplus(grids, kernel_stencils) -> Callable:
    """Create closure for computing grid contribution to the energy"""
    anterpolate_level_one = create_anterpolation_operator(grids[1])
    compute_gridpotential_level_one = create_compute_gridpotential_level_one(
        grids=grids, kernel_stencils=kernel_stencils
    )

    def compute_U_oneplus(
        positions: jax.Array, charges: jax.Array
    ) -> jax.Array:
        gridcharge_level_one = anterpolate_level_one(
            positions_1d=positions, charges=charges
        )
        gridpotential_level_one = compute_gridpotential_level_one(
            gridcharge_level_one
        )

        # TODO: splinevals and indices from anterpolation could, in principle,
        #  be reused here instead of recalculated;
        #  but would this be any faster in practice?
        splinevals, indices = grids[1].evaluate_bspline_basis_multi(positions)

        return 0.5 * jnp.sum(
            charges
            * (gridpotential_level_one[indices] * splinevals).sum(axis=1)
        )

    return compute_U_oneplus


def create_compute_U_and_f_oneplus(grids, kernel_stencils) -> Callable:
    """Create closure for computing grid contribution to energy and forces"""
    anterpolate_level_one = create_anterpolation_operator(grids[1])
    compute_gridpotential_level_one = create_compute_gridpotential_level_one(
        grids=grids, kernel_stencils=kernel_stencils
    )

    def compute_U_and_f_oneplus(
        positions: jax.Array, charges: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        gridcharge_level_one = anterpolate_level_one(
            positions_1d=positions, charges=charges
        )
        gridpotential_level_one = compute_gridpotential_level_one(
            gridcharge_level_one
        )

        # TODO: splinevals, splinegrads and indices from anterpolation could,
        #  in principle, be reused here instead of recalculated;
        #  but would this be any faster in practice?
        splinevals, indices = grids[1].evaluate_bspline_basis_multi(positions)
        splinegrads, _ = grids[1].evaluate_bspline_basis_gradient_multi(
            positions
        )
        energy = 0.5 * jnp.sum(
            charges
            * (gridpotential_level_one[indices] * splinevals).sum(axis=1)
        )
        forces = -charges * jnp.sum(
            gridpotential_level_one[indices] * splinegrads, axis=1
        )

        return energy, forces

    return compute_U_and_f_oneplus
