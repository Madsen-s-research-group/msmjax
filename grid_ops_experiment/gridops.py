from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt

from msmjax.bspline_basis import create_bspline_basis_element


class GridAxis1D(NamedTuple):
    periodic: bool
    length: float
    h: float
    n_domain: int
    n_total: int
    process_raw_indices: Callable
    wrap_indices_if_periodic: Callable
    wrap_or_invalidate_indices: Callable
    evaluate_bspline_basis_multi: Callable
    evaluate_bspline_basis_gradient_multi: Callable


def set_up_grid_axis(length: float, h: float, p: int, periodic: bool):
    if p % 2 != 0:
        raise ValueError("p must be even")

    if periodic:
        # TODO
        # raise ValueError("Periodic axes not supported yet.")
        n_domain = int(onp.ceil(length / h))
        n_total = n_domain
    else:
        # TODO: determination of number of grid points might not be numerically robust
        n_domain = int(onp.ceil(length / h)) + 1
        n_total = n_domain + p

    def process_raw_indices_nonperiodic(raw_indices: npt.ArrayLike):
        return raw_indices + p // 2

    def process_raw_indices_periodic(raw_indices: npt.ArrayLike):
        # raise  # TODO
        return raw_indices

    if periodic:
        process_raw_indices = process_raw_indices_periodic
    else:
        process_raw_indices = process_raw_indices_nonperiodic

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
            # TODO: Checking only for negative indices would be enough, but is it less readable?
            is_in_bounds = jnp.logical_and(indices >= 0, indices < n_total)
            return jnp.where(
                is_in_bounds, indices, intentionally_out_of_bounds_index
            )

    def evaluate_bspline_basis_for_one_particle(
        x: float,
    ) -> Tuple[jax.Array, jax.Array]:
        x_over_h = x / h
        reference_index = jnp.ceil(x_over_h).astype(int)
        raw_indices = reference_index + jnp.arange(-p // 2, p // 2)
        splinevals = jax.vmap(bspline_basis_element)(x_over_h - raw_indices)
        # TODO: should this return raw or processed indices?
        indices = wrap_indices_if_periodic(process_raw_indices(raw_indices))

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

    return GridAxis1D(
        periodic=periodic,
        length=length,
        h=h,
        n_domain=n_domain,
        n_total=n_total,
        process_raw_indices=process_raw_indices,
        wrap_indices_if_periodic=wrap_indices_if_periodic,
        wrap_or_invalidate_indices=wrap_or_invalidate_indices,
        evaluate_bspline_basis_multi=evaluate_bspline_basis_multi,
        evaluate_bspline_basis_gradient_multi=evaluate_bspline_basis_gradient_multi,
    )


def create_anterpolation_function(grid: GridAxis1D):
    def anterpolate(positions_1d: jax.Array, charges: jax.Array) -> jax.Array:
        splinevals, indices = grid.evaluate_bspline_basis_multi(positions_1d)
        gridcharge = jnp.zeros(grid.n_total)
        gridcharge = gridcharge.at[indices].add(
            charges[:, jnp.newaxis] * splinevals
        )

        return gridcharge

    return anterpolate


def make_restriction_operator(
    grid_source: GridAxis1D, grid_target: GridAxis1D, J: npt.ArrayLike
) -> Callable:
    # TODO: full J or J_zeroplus part only?
    J = jnp.asarray(J)

    # TODO: How should p be passed/inferred? Probably should be an attribute of the grid?
    p = len(J) - 1

    def get_gridinds_one_below(m_raw: int) -> jax.Array:
        # TODO: would 2 * m_raw + jnp.arange(-p // 2, p // 2 + 1) be better readable (assuming that is really correct and the same)?
        n_raw_selected = (2 * m_raw - p // 2) + jnp.arange(p + 1)
        n_selected = grid_source.process_raw_indices(n_raw_selected)
        return n_selected

    if grid_target.periodic:
        # raise  # TODO: not implemented
        raw_ms = jnp.arange(grid_target.n_total)
        ms = grid_target.process_raw_indices(raw_ms)
    else:
        # TODO: should this be handled by a generic grid function as well? (how?)
        raw_ms = jnp.arange(grid_target.n_total) - p // 2
        ms = grid_target.process_raw_indices(raw_ms)

    def restrict(array_fine: jax.Array) -> jax.Array:
        selected_ns = jax.vmap(get_gridinds_one_below)(raw_ms)
        selected_ns = grid_source.wrap_or_invalidate_indices(selected_ns)
        selected_gridcharges_below = array_fine.at[selected_ns].get(
            mode="fill", fill_value=0.0
        )

        array_coarse = jnp.zeros(grid_target.n_total)
        array_coarse = array_coarse.at[ms].add(
            (selected_gridcharges_below * J).sum(axis=1)
        )

        return array_coarse

    return restrict


def make_prolongation_operator(
    grid_source: GridAxis1D,
    grid_target: GridAxis1D,
    p: int,
    J_zeroplus: npt.ArrayLike,
):
    # TODO: J (pass full or only zeroplus part?)
    J_zeroplus = jnp.asarray(J_zeroplus)

    start_even = onp.ceil(onp.round(-p / 4, decimals=1)).astype(int)
    end_even = onp.floor(onp.round(p / 4, decimals=1)).astype(int)
    start_odd = onp.ceil(onp.round(0.5 - p / 4, decimals=1)).astype(int)
    end_odd = onp.floor(onp.round(0.5 + p / 4, decimals=1)).astype(int)

    dists_to_neighboring_ms_even = jnp.arange(start_even, end_even + 1)
    dist_to_neighboring_ms_odd = jnp.arange(start_odd, end_odd + 1)

    inds_into_J_even = -2 * dists_to_neighboring_ms_even
    inds_into_J_odd = 1 - 2 * dist_to_neighboring_ms_odd

    def get_ns_one_above_even(m_raw: int):
        n_raw_selected = m_raw // 2 + dists_to_neighboring_ms_even
        return grid_source.process_raw_indices(n_raw_selected)

    def get_ns_one_above_odd(m_raw: int):
        n_raw_selected = m_raw // 2 + dist_to_neighboring_ms_odd
        return grid_source.process_raw_indices(n_raw_selected)

    if grid_target.periodic:
        # raise  # TODO: not implemented
        raw_ms = jnp.arange(grid_target.n_total)
        ms = grid_target.process_raw_indices(raw_ms)
        slice_even = slice(0, None, 2)
        slice_odd = slice(1, None, 2)
    else:
        # TODO: should this be handled by a generic grid function as well? (how?)
        raw_ms = jnp.arange(grid_target.n_total) - p // 2
        ms = grid_target.process_raw_indices(raw_ms)
        slice_even = slice((p // 2) % 2, None, 2)
        slice_odd = slice(1 - (p // 2) % 2, None, 2)

    def prolongate(array_coarse: jax.Array) -> jax.Array:
        ns_even_ms = jax.vmap(get_ns_one_above_even)(raw_ms[slice_even])
        ns_odd_ms = jax.vmap(get_ns_one_above_odd)(raw_ms[slice_odd])

        ns_even_ms = grid_source.wrap_indices_if_periodic(ns_even_ms)
        ns_odd_ms = grid_source.wrap_indices_if_periodic(ns_odd_ms)

        array_fine = jnp.zeros(grid_target.n_total)
        array_fine = array_fine.at[ms[slice_even]].add(
            (
                array_coarse[ns_even_ms]
                * J_zeroplus[jnp.abs(inds_into_J_even)]
            ).sum(axis=1)
        )
        array_fine = array_fine.at[ms[slice_odd]].add(
            (
                array_coarse[ns_odd_ms] * J_zeroplus[jnp.abs(inds_into_J_odd)]
            ).sum(axis=1)
        )

        return array_fine

    return prolongate


def create_interaction_operator(
    grid: GridAxis1D, kernelstencil: npt.ArrayLike
):
    kernelstencil = jnp.asarray(kernelstencil)
    interaction_range = len(kernelstencil) // 2
    gridsize = grid.n_total

    def apply_interaction(inarray):
        ms = jnp.arange(gridsize)
        ns_within_kernel_range = ms[:, jnp.newaxis] + jnp.arange(
            -interaction_range, interaction_range + 1
        )

        # TODO: The whole bounds-checking step could be put into a new,
        #  more general, function for handling boundary conditions?
        # TODO: Current implementation works for non-periodic case only.
        selected_ns = grid.wrap_or_invalidate_indices(ns_within_kernel_range)
        selected_values = inarray.at[selected_ns].get(
            mode="fill", fill_value=0.0
        )

        outarray = jnp.zeros(gridsize)
        outarray = outarray.at[ms].add(
            (selected_values * kernelstencil).sum(axis=1)
        )

        return outarray

    return apply_interaction


def create_compute_gridpotential_level_one(
    grids,
    kernelstencils,
    J: npt.ArrayLike,
) -> Callable:
    J_zeroplus = J[len(J) // 2 :]  # TODO: pass J or J_zeroplus?
    p = len(J) - 1  # TODO: How should p be passed/inferred?

    max_gridlevel = len(grids) - 1

    anterpolate = create_anterpolation_function(grids[1])

    restriction_funcs = {}
    for lvl in range(2, max_gridlevel + 1):
        restrict = make_restriction_operator(
            grid_source=grids[lvl - 1], grid_target=grids[lvl], J=J
        )
        restriction_funcs[lvl] = restrict

    prolongation_funcs = {}
    for lvl in range(1, max_gridlevel):
        prolongate = make_prolongation_operator(
            grid_source=grids[lvl + 1],
            grid_target=grids[lvl],
            J_zeroplus=J_zeroplus,
            p=p,
        )
        prolongation_funcs[lvl] = prolongate

    interaction_funcs = {}
    for lvl in range(1, max_gridlevel + 1):
        interact = create_interaction_operator(
            grid=grids[lvl], kernelstencil=kernelstencils[lvl]
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


def make_compute_U_oneplus(
    grids, kernelstencils, J: npt.ArrayLike
) -> Callable:
    compute_gridpotential_level_one = create_compute_gridpotential_level_one(
        grids=grids, kernelstencils=kernelstencils, J=J
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
        return (
            0.5
            * (
                charges[:, jnp.newaxis]
                * gridpotential_level_one[indices]
                * splinevals
            ).sum()
        )

    return compute_U_oneplus


def make_compute_U_and_f_oneplus(
    grids, kernelstencils, J: npt.ArrayLike
) -> Callable:
    compute_gridpotential_level_one = create_compute_gridpotential_level_one(
        grids=grids,
        kernelstencils=kernelstencils,
        J=J,
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
        U_0 = (
            0.5
            * (
                charges[:, jnp.newaxis]
                * gridpotential_level_one[indices]
                * splinevals
            ).sum()
        )
        f_0 = -charges * (gridpotential_level_one[indices] * splinegrads).sum(
            axis=1
        )

        return U_0, f_0

    return compute_U_and_f_oneplus
