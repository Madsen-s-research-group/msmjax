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
