from typing import Callable, NamedTuple

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
    get_gridindices_and_splinevals: Callable


def set_up_grid_axis(length: float, h: float, p: int, periodic: bool):
    if p % 2 != 0:
        raise ValueError("p must be even")

    if periodic:
        raise ValueError("Periodic axes not supported yet.")  # TODO
    else:
        # TODO: determination of number of grid points might not be numerically robust
        n_domain = int(onp.ceil(length / h)) + 1
        n_total = n_domain + p

    def process_raw_indices_nonperiodic(raw_indices: npt.ArrayLike):
        return raw_indices + p // 2

    def process_raw_indices_periodic(raw_indices: npt.ArrayLike):
        raise  # TODO

    if periodic:
        process_raw_indices = process_raw_indices_periodic
    else:
        process_raw_indices = process_raw_indices_nonperiodic

    bspline_basis_element = create_bspline_basis_element(order=p - 1)

    def get_gridindices_and_splinevals(x: float):
        x_over_h = x / h
        reference_index = jnp.ceil(x_over_h).astype(int)
        raw_indices = reference_index + jnp.arange(-p // 2, p // 2)
        splinevals = jax.vmap(bspline_basis_element)(x_over_h - raw_indices)
        # TODO: should this return raw or processed indices?
        return process_raw_indices(raw_indices), splinevals

    return GridAxis1D(
        periodic=periodic,
        length=length,
        h=h,
        n_domain=n_domain,
        n_total=n_total,
        process_raw_indices=process_raw_indices,
        get_gridindices_and_splinevals=get_gridindices_and_splinevals,
    )


def create_anterpolation_function(grid: GridAxis1D):
    vmapped_get_gridindices_and_splinevals = jax.vmap(
        grid.get_gridindices_and_splinevals
    )

    def anterpolate(positions_1d, charges):
        indices, splinevals = vmapped_get_gridindices_and_splinevals(
            positions_1d
        )

        gridcharge = jnp.zeros(grid.n_total)
        gridcharge = gridcharge.at[indices].add(
            charges[:, jnp.newaxis] * splinevals
        )

        return gridcharge, indices, splinevals

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
        raise  # TODO: not implemented
    else:
        # TODO: should this be handled by a generic grid function as well? (how?)
        raw_ms = jnp.arange(grid_target.n_total) - p // 2
        ms = grid_target.process_raw_indices(raw_ms)

    def restrict(array_fine: jax.Array) -> jax.Array:
        selected_ns = jax.vmap(get_gridinds_one_below)(raw_ms)
        is_in_bounds = jnp.logical_and(
            selected_ns >= 0, selected_ns < grid_source.n_total
        )
        intentionally_out_of_bounds_index = grid_source.n_total
        selected_ns = jnp.where(
            is_in_bounds, selected_ns, intentionally_out_of_bounds_index
        )
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
        raise  # TODO: not implemented
        # slice_even = slice(0, None, 2)
        # slice_odd = slice(1, None, 2)
    else:
        # TODO: should this be handled by a generic grid function as well? (how?)
        raw_ms = jnp.arange(grid_target.n_total) - p // 2
        ms = grid_target.process_raw_indices(raw_ms)
        slice_even = slice((p // 2) % 2, None, 2)
        slice_odd = slice(1 - (p // 2) % 2, None, 2)

    def prolongate(array_coarse: jax.Array) -> jax.Array:
        # TODO: the identification of odd/even is wrong when p // 2 is odd!
        ns_even_ms = jax.vmap(get_ns_one_above_even)(raw_ms[slice_even])
        ns_odd_ms = jax.vmap(get_ns_one_above_odd)(raw_ms[slice_odd])

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
        is_in_bounds = jnp.logical_and(
            ns_within_kernel_range >= 0, ns_within_kernel_range < gridsize
        )
        intentionally_out_of_bounds_index = gridsize
        selected_ns = jnp.where(
            is_in_bounds,
            ns_within_kernel_range,
            intentionally_out_of_bounds_index,
        )

        selected_values = inarray.at[selected_ns].get(
            mode="fill", fill_value=0.0
        )

        outarray = jnp.zeros(gridsize)
        outarray = outarray.at[ms].add(
            (selected_values * kernelstencil).sum(axis=1)
        )

        return outarray

    return apply_interaction
