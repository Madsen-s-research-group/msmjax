#!/usr/bin/env python
"""B-spline basis function implementation"""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from msmjax.utils.misc import _divide_zero_safe


def characteristic(knots: jnp.ndarray, eval_point: ArrayLike) -> jnp.ndarray:
    """Characteristic function chi on intervals defined by knots.
    chi(eval_point) evaluates to 1 for knots[t] <= eval_point < knots[t+1],
    0 otherwise

    Args:
        knots: Array containing the positions of the knots.
            Have to be sorted monotonically increasing
        eval_point: Point to evaluate the characteristic basis at
    Returns:
        Array containing characteristic function evaluation chi(eval_point)
        on each interval. Array is of length len(knots) - 1
    """
    return jnp.where(
        jnp.logical_and(
            knots[:-1] <= eval_point,
            eval_point < knots[1:],
        ),
        1.0,
        0.0,
    )


def evaluate_basis_element(
    knots: jnp.ndarray,
    eval_point: ArrayLike,
) -> jnp.ndarray:
    """Evaluate the B-spline basis on the reference interval defined by knots
    using the recursive definition by de Boor. The order of the basis element
    is given by len(knots) - 2. For more information:
    "A Practical Guide to Splines", C. de Boor, Springer, 2001.

    Args:
        knots: Array containing the positions of the knots.
            Array has to be sorted monotonically increasing
        eval_point: Point to evaluate the B-spline basis at
    Returns:
        B-spline basis evaluation for the given reference interval
    """

    def de_boor_scan_step(
        evals: jnp.ndarray,
        current_order: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, None]:
        prefactor_first = _divide_zero_safe(
            eval_minus_knots,
            jnp.roll(knots, -current_order) - knots,
        )
        knots_shifted_upmost = jnp.roll(knots, -(current_order + 1))
        prefactor_second = _divide_zero_safe(
            knots_shifted_upmost - eval_point,
            knots_shifted_upmost - jnp.roll(knots, -1),
        )
        return (
            prefactor_first[:-1] * evals
            + prefactor_second[:-1] * jnp.roll(evals, -1)
        ), None

    eval_minus_knots = eval_point - knots  # Constant for every iteration
    # "0th order" splines are characteristics -> init, iterate upwards
    result, _ = jax.lax.scan(
        f=de_boor_scan_step,
        init=characteristic(knots, eval_point),
        xs=jnp.arange(1, knots.shape[0] - 1),
    )
    return result[0]


def create_bspline_basis_element(
    order: int = 3,
) -> Callable[[ArrayLike], jnp.ndarray]:
    """Wrapper to create a B-spline basis element of given order centered at 0

    Args:
        order: Order of the spline, has to be greater or equal to 1.
            Defines the reference interval [-(order + 1) / 2, (order + 1) / 2]
    Returns:
        Function that takes the point on the reference interval as input
        and returns the evaluation of the B-spline basis at that point
    """
    if order < 1:
        raise ValueError("Requested order is smaller than 1.")
    knots = jnp.arange(-(order + 1) / 2, (order + 3) / 2)
    return partial(evaluate_basis_element, knots)
