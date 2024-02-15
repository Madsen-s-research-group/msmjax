from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt

from msmjax.bspline_basis import create_bspline_basis_element


class BSplineInterpolationGrid1D(NamedTuple):
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

    return BSplineInterpolationGrid1D(
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
        evaluate_bspline_basis_multi=evaluate_bspline_basis_multi,
        evaluate_bspline_basis_gradient_multi=evaluate_bspline_basis_gradient_multi,
    )


def create_anterpolation_operator(grid: BSplineInterpolationGrid1D):
    """Create a function that anterpolates charge from particles to grid"""

    def anterpolate(positions_1d: jax.Array, charges: jax.Array) -> jax.Array:
        """Anterpolate charge from particles to grid"""
        splinevals, indices = grid.evaluate_bspline_basis_multi(positions_1d)
        gridcharge = jnp.zeros(grid.n_total)
        gridcharge = gridcharge.at[indices].add(
            charges[:, jnp.newaxis] * splinevals
        )

        return gridcharge

    return anterpolate


def create_restriction_operator(
    grid_source_fine: BSplineInterpolationGrid1D,
    grid_target_coarse: BSplineInterpolationGrid1D,
) -> Callable:
    """Create function that performs the restriction operation"""
    # TODO: check if both grids have same J and p?
    # TODO: check if shape of J is compatible with p?
    p = grid_source_fine.p
    J_zeroplus = grid_source_fine.J_zeroplus
    J = jnp.concatenate((J_zeroplus[::-1][:-1], J_zeroplus))

    def get_neigbhor_inds_on_sourcegrid(idx_targetgrid: int) -> jax.Array:
        """Get a target-grid index's neighbor indices on source grid."""
        raw_idx_targetgrid = grid_target_coarse.to_raw_indices(idx_targetgrid)
        raw_neighbor_inds_sourcegrid = 2 * raw_idx_targetgrid + jnp.arange(
            -p // 2, p // 2 + 1
        )
        neighbor_inds_sourcegrid = grid_source_fine.from_raw_indices(
            raw_neighbor_inds_sourcegrid
        )
        return neighbor_inds_sourcegrid

    inds_targetgrid = jnp.arange(grid_target_coarse.n_total)

    def restrict(in_array_fine: jax.Array) -> jax.Array:
        """Restrict array defined on grid to the next-coarser (higher) grid"""
        neighbor_inds_sourcegrid = jax.vmap(get_neigbhor_inds_on_sourcegrid)(
            inds_targetgrid
        )
        neighbor_inds_sourcegrid = grid_source_fine.wrap_or_invalidate_indices(
            neighbor_inds_sourcegrid
        )
        neighbor_values_sourcegrid = in_array_fine.at[
            neighbor_inds_sourcegrid
        ].get(mode="fill", fill_value=0.0)

        out_array_coarse = jnp.zeros(grid_target_coarse.n_total)
        out_array_coarse = out_array_coarse.at[inds_targetgrid].add(
            (neighbor_values_sourcegrid * J).sum(axis=1)
        )

        return out_array_coarse

    return restrict


def make_prolongation_operator(
    grid_source_coarse: BSplineInterpolationGrid1D,
    grid_target_fine: BSplineInterpolationGrid1D,
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
    grid: BSplineInterpolationGrid1D, kernel_stencil: npt.ArrayLike
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


def create_compute_gridpotential_level_one(grids, kernel_stencils) -> Callable:
    # TODO: check if all grids have same J and p?
    # TODO: check if shape of J is compatible with p?

    max_gridlevel = len(grids) - 1

    anterpolate = create_anterpolation_operator(grids[1])

    restriction_funcs = {}
    for lvl in range(2, max_gridlevel + 1):
        restrict = create_restriction_operator(
            grid_source_fine=grids[lvl - 1], grid_target_coarse=grids[lvl]
        )
        restriction_funcs[lvl] = restrict

    prolongation_funcs = {}
    for lvl in range(1, max_gridlevel):
        prolongate = make_prolongation_operator(
            grid_source_coarse=grids[lvl + 1], grid_target_fine=grids[lvl]
        )
        prolongation_funcs[lvl] = prolongate

    interaction_funcs = {}
    for lvl in range(1, max_gridlevel + 1):
        interact = create_interaction_operator(
            grid=grids[lvl], kernel_stencil=kernel_stencils[lvl]
        )
        interaction_funcs[lvl] = interact

    def compute_gridpotential_level_one(
        positions: jax.Array, charges: jax.Array
    ) -> jax.Array:
        # Compute lowest-level grid charge from particle positions and charges
        gridcharge_level_one = anterpolate(
            positions_1d=positions, charges=charges
        )
        gridcharges_all_levels = {1: gridcharge_level_one}

        # Go up ladder
        for lvl in range(2, max_gridlevel + 1):
            restrict = restriction_funcs[lvl]
            gridcharge_fine = gridcharges_all_levels[lvl - 1]
            gridcharge_coarse = restrict(gridcharge_fine)
            gridcharges_all_levels[lvl] = gridcharge_coarse

        # Apply top-level interaction
        gridcharge_toplevel = gridcharges_all_levels[max_gridlevel]
        interaction_toplevel = interaction_funcs[max_gridlevel]
        gridpotential = interaction_toplevel(gridcharge_toplevel)

        # Go down ladder
        for lvl in range(max_gridlevel - 1, 0, -1):
            gridpotential = interaction_funcs[lvl](
                gridcharges_all_levels[lvl]
            ) + prolongation_funcs[lvl](gridpotential)

        return gridpotential

    return compute_gridpotential_level_one


def make_compute_U_oneplus(grids, kernelstencils) -> Callable:
    compute_gridpotential_level_one = create_compute_gridpotential_level_one(
        grids=grids, kernel_stencils=kernelstencils
    )

    def compute_U_oneplus(
        positions: jax.Array, charges: jax.Array
    ) -> jax.Array:
        gridpotential_level_one = compute_gridpotential_level_one(
            positions=positions, charges=charges
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


def make_compute_U_and_f_oneplus(grids, kernelstencils) -> Callable:
    compute_gridpotential_level_one = create_compute_gridpotential_level_one(
        grids=grids, kernel_stencils=kernelstencils
    )

    def compute_U_and_f_oneplus(
        positions: jax.Array, charges: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        gridpotential_level_one = compute_gridpotential_level_one(
            positions=positions, charges=charges
        )
        # TODO: splinevals, splinegrads and indices from anterpolation could,
        #  in principle, be reused here instead of recalculated;
        #  but would this be any faster in practice?
        splinevals, indices = grids[1].evaluate_bspline_basis_multi(positions)
        splinegrads, _ = grids[1].evaluate_bspline_basis_gradient_multi(
            positions
        )
        U_0 = 0.5 * jnp.sum(
            charges
            * (gridpotential_level_one[indices] * splinevals).sum(axis=1)
        )
        f_0 = -charges * jnp.sum(
            gridpotential_level_one[indices] * splinegrads, axis=1
        )

        return U_0, f_0

    return compute_U_and_f_oneplus
