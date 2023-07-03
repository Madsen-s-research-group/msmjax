#!/usr/bin/env python
"""B-spline basis function implementation"""
from functools import partial
from typing import Callable

import jax.numpy as jnp


def _divide_zero_safe(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:
    """Function that forces the result of dividing by 0 to be equal to 0.0
    in a jit- and autodiff-compatible way

    Args:
        numerator: Values in the numerator
        denominator: Values in the denominator, may contain zeros
    Returns:
        numerator / denominator with result == 0.0 where denominator == 0.0
    """
    denominator_masked = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(
        denominator == 0.0,
        0.0,
        numerator / denominator_masked,
    )


def characteristic(knots: jnp.ndarray, eval_point: float) -> jnp.ndarray:
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
    return jnp.where(jnp.logical_and(
        knots[:-1] <= eval_point,
        eval_point < knots[1:],
    ), 1.0, 0.0)


# Both knots and order are passed since this function could be adapted for
# evaluation of more than one basis element (return array instead of scalar)
def evaluate_basis(
    knots: jnp.ndarray,
    order: int,
    eval_point: float,
) -> jnp.ndarray:
    """Evaluate the B-spline basis on the reference interval defined by knots
    using the recursive definition by de Boor.
    (See "A Practical Guide to Splines", C. de Boor, 1978 for more information)

    Args:
        knots: Array containing the positions of the knots.
            Array has to be sorted monotonically increasing
        order: The order of the spline
        eval_point: Point to evaluate the B-spline basis at
    Returns:
        B-spline basis evaluation for the given reference interval
    """
    # "0th order" splines are characteristic functions
    result = characteristic(knots, eval_point)
    eval_minus_knots = eval_point - knots  # Constant for every iteration
    for k in range(1, order + 1):
        prefactor_first = _divide_zero_safe(
            eval_minus_knots,
            jnp.roll(knots, -k) - knots,
        )
        knots_shifted_upmost = jnp.roll(knots, -(k + 1))
        prefactor_second = _divide_zero_safe(
            knots_shifted_upmost - eval_point,
            knots_shifted_upmost - jnp.roll(knots, -1)
        )
        result = (prefactor_first[:-1] * result +
                  prefactor_second[:-1] * jnp.roll(result, -1))
    return result[0]


def create_bspline_basis(
    order: int,
) -> Callable[[float], jnp.ndarray]:
    """
    Create a function which evaluates the B-spline basis for the given order.
    Returned function is compatible with jax.jit and automatic differentiation

    Args:
        order: Order of the spline.
            Defines the reference interval [-(order + 1) / 2, (order + 3) / 2]
    Returns:
        Function that takes the point on the reference interval as input
        and returns the evaluation of the B-spline basis at that point
    """
    knots = jnp.arange(-(order + 1) / 2, (order + 3) / 2)
    return partial(evaluate_basis, knots, order)


if __name__ == "__main__":
    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BSpline
    from scipy.signal import bspline

    ORDER = 3
    PLOT = True

    basis = create_bspline_basis(ORDER)
    eval_points = jnp.linspace(-2.0, 2.0, num=51)

    evaluations, gradients = jax.jit(jax.vmap(jax.value_and_grad(
        basis
    )))(eval_points)
    basis_knots = np.arange(-(ORDER + 1) / 2, (ORDER + 3) / 2)
    evaluations_scipy = BSpline.basis_element(basis_knots)(eval_points)
    evaluations_scipy[
        (eval_points < basis_knots[0]) | (eval_points > basis_knots[-1])
    ] = 0.0
    evaluations_scipy_old = bspline(eval_points, ORDER)
    print(
        "JAX implementation and current scipy BSpline all close? "
        f"{jnp.allclose(evaluations, evaluations_scipy)}"
    )
    print(
        "JAX implementation and deprecated scipy bspline all close? "
        f"{jnp.allclose(evaluations, evaluations_scipy_old)}"
    )

    if PLOT:
        fig, ax = plt.subplots(1, 1)
        ax.plot(eval_points, evaluations, label="Value")
        ax.plot(eval_points, gradients, label="Gradient")
        ax.legend()
        ax.set_title(f"B-Spline basis element of order {ORDER}")
        fig.show()
