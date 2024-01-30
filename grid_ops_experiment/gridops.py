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


def make_restriction_function(
    grid: GridAxis1D, grid_below: GridAxis1D, J: npt.ArrayLike
) -> Callable:
    J = jnp.asarray(J)

    # TODO: How should p be passed/inferred? Probably should be an attribute of the grid?
    p = len(J) - 1

    def get_gridinds_one_below(m_raw: int) -> jax.Array:
        # TODO: would 2 * m_raw + jnp.arange(-p // 2, p // 2 + 1) be better readable (assuming that is really correct and the same)?
        n_raw_selected = (2 * m_raw - p // 2) + jnp.arange(p + 1)
        n_selected = grid_below.process_raw_indices(n_raw_selected)
        return n_selected

    if grid.periodic:
        raise  # TODO: not implemented
    else:
        # TODO: should this be handled by a generic grid function as well? (how?)
        raw_ms = jnp.arange(grid.n_total) - p // 2
        ms = grid.process_raw_indices(raw_ms)

    def apply_restrict(gridcharge_below: jax.Array) -> jax.Array:
        selected_ns = jax.vmap(get_gridinds_one_below)(raw_ms)
        is_in_bounds = jnp.logical_and(
            selected_ns >= 0, selected_ns < grid_below.n_total
        )
        intentionally_out_of_bounds_index = grid_below.n_total + 1
        selected_ns = jnp.where(
            is_in_bounds, selected_ns, intentionally_out_of_bounds_index
        )
        selected_gridcharges_below = gridcharge_below.at[selected_ns].get(
            mode="fill", fill_value=0.0
        )

        gridcharge = jnp.zeros(grid.n_total)
        gridcharge = gridcharge.at[ms].add(
            (selected_gridcharges_below * J).sum(axis=1)
        )

        return gridcharge

    return apply_restrict


def make_prolongation_operator(
    grid: GridAxis1D, grid_above: GridAxis1D, p: int, J_zeroplus: npt.ArrayLike
):
    # TODO: J
    J_zeroplus = jnp.asarray(J_zeroplus)

    start_even = onp.ceil(onp.round(-p / 4, decimals=1)).astype(int)
    end_even = onp.floor(onp.round(p / 4, decimals=1)).astype(int)
    start_odd = onp.ceil(onp.round(0.5 - p / 4, decimals=1)).astype(int)
    end_odd = onp.floor(onp.round(0.5 + p / 4, decimals=1)).astype(int)

    displacements_even = jnp.arange(start_even, end_even + 1)
    displacements_odd = jnp.arange(start_odd, end_odd + 1)

    inds_into_J_even = -2 * displacements_even
    inds_into_J_odd = 1 - 2 * displacements_odd

    def get_ns_one_above_even(m_raw: int):
        n_raw_selected = m_raw // 2 + displacements_even
        return grid_above.process_raw_indices(n_raw_selected)

    def get_ns_one_above_odd(m_raw: int):
        n_raw_selected = m_raw // 2 + displacements_odd
        return grid_above.process_raw_indices(n_raw_selected)

    if grid.periodic:
        raise  # TODO: not implemented
    else:
        # TODO: should this be handled by a generic grid function as well? (how?)
        raw_ms = jnp.arange(grid.n_total) - p // 2
        ms = grid.process_raw_indices(raw_ms)
        if (p // 2) % 2 == 0:
            slice_even = slice(0, None, 2)
            slice_odd = slice(1, None, 2)
        else:
            slice_even = slice(1, None, 2)
            slice_odd = slice(0, None, 2)

    def prolongate(gridarray_above: jax.Array) -> jax.Array:
        gridarray = jnp.zeros(grid.n_total)

        # TODO: the identification of odd/even is wrong when p // 2 is odd!
        ns_even_ms = jax.vmap(get_ns_one_above_even)(raw_ms[slice_even])
        ns_odd_ms = jax.vmap(get_ns_one_above_odd)(raw_ms[slice_odd])

        gridarray = gridarray.at[ms[slice_even]].add(
            (
                gridarray_above[ns_even_ms]
                * J_zeroplus[jnp.abs(inds_into_J_even)]
            ).sum(axis=1)
        )
        gridarray = gridarray.at[ms[slice_odd]].add(
            (
                gridarray_above[ns_odd_ms]
                * J_zeroplus[jnp.abs(inds_into_J_odd)]
            ).sum(axis=1)
        )

        return gridarray

    return prolongate
