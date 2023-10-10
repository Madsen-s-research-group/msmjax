"""Functionality for splitting interaction kernels into sum of partial kernels.

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp


class SofteningFunctionOneOverR:
    """Class for constructing and evaluating softener for the 1/r kernel.

    The softener is a function of a dimensionless argument rho that is equal
    to 1/rho for rho >= 1 and bounded and smooth for rho < 1.

    Args:
        order: Order (as a function of s = rho**2) of the Taylor polynomial
            that the softening function consists of for rho < 1.
            In line with common spline terminology, its polynomial degree (as
            a function of s = rho**2) is equal to order - 1.
    """

    def __init__(self, order: int):
        if not isinstance(order, (int, onp.integer, jnp.integer)):
            raise ValueError("'order' must be an integer.")
        if order < 1:
            raise ValueError("The expansion must have at least one term.")
        # TODO: Should it be checked (at this point) whether the order is even?
        self.order = order
        last_coeff = 1.0
        coeffs = [last_coeff]
        for i_term in range(1, self.order):
            last_coeff *= (1.0 / i_term - 2) / 2.0
            coeffs.append(last_coeff)
        coeffs = list(reversed(coeffs))
        self.coeffs = jnp.array(coeffs)

    def __call__(self, rho):
        return jnp.where(
            rho < 1.0, jnp.polyval(self.coeffs, rho * rho - 1.0), 1.0 / rho
        )


def split_one_over_r_kernel(
    max_level: int, level_zero_cutoff: float, softening_function: Callable
):
    """Split kernel 1/r in (max_level + 1) terms according to reference.

    The splitting terms sum up to the Coulomb kernel 1/r like this:
    1/r = g_0(r) + g_1(r) + g_2(r) + ... + g_{max_order}(r)

    Args:
        max_level: The number of splits to be performed.
        level_zero_cutoff: The cutoff radius of the level-zero kernel function.
        softening_function: The basic smoothing function for this splitting.

    Returns:
        A list of one-argument functions g_l with l from zero to `max_order`
        that represent the terms in the splitting of the interaction kernel.
        These have their respective cutoffs 'built in' already (in the sense
        that they evaluate to zero for distances beyond) and take their
        arguments in the same length units that `level_zero_cutoff` was
        supplied in.

    Raises:
        ValueError: If the arguments do not make sense.
    """
    # TODO: check max_level positive
    if not isinstance(max_level, (int, onp.integer, jnp.integer)):
        raise ValueError("'max_level' must be an integer.")
    if max_level < 1:
        raise ValueError(
            "'max_level' must be at least one (which corresponds to the case "
            "of splitting the kernel in two terms)."
        )
    if level_zero_cutoff <= 0.0:
        raise ValueError("Cutoff must be positive.")

    def gamma_0(rho):
        return 1.0 / rho - softening_function(rho)

    def gamma_l(rho):
        return 2.0 * softening_function(2.0 * rho) - softening_function(rho)

    def gamma_L(rho):
        return 2.0 * softening_function(2.0 * rho)

    def g_l_factory(a_l, gamma):
        def g_l(r):
            nonlocal a_l, gamma
            return gamma(r / a_l) / a_l

        return g_l

    all_gammas = [gamma_0] + [gamma_l] * (max_level - 1) + [gamma_L]

    nruter = []
    a_l = level_zero_cutoff
    for ell, gamma in enumerate(all_gammas):
        nruter.append(g_l_factory(a_l, gamma))
        a_l *= 2.0

    return nruter


if __name__ == "__main__":
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import time

    import matplotlib.pyplot as plt

    softening_function = SofteningFunctionOneOverR(4)
    softening_function_jitted = jax.jit(softening_function)
    softening_function_vmapped = jax.jit(jax.vmap(softening_function))

    dgamma = jax.grad(softening_function)

    print(dgamma(1.0e-8))
    print(dgamma(1.0))

    fig, ax = plt.subplots()
    rhos = onp.linspace(0, 5, 501)
    ax.plot(rhos, softening_function(rhos))
    ax.plot(rhos, 1.0 / rhos, linestyle="--")
    ax.axvline(1, linestyle=":", color="gray")
    ax.set_ylim((0.0, softening_function(rhos).max() + 0.25))
    plt.show()

    print("coeffs:")
    print(softening_function.coeffs)
    print()

    kernels = split_one_over_r_kernel(
        max_level=1,
        level_zero_cutoff=3.0,
        softening_function=softening_function,
    )
    print("len(kernels):")
    print(len(kernels))
    print()

    # print("- brute force:")
    # t1 = time.time()
    # for x in jnp.linspace(0.5, 1.0, 5001):
    #     _ = softening_function(x).block_until_ready()
    # t2 = time.time()
    # print(t2 - t1)
    # print()
    #
    # print("- jit:")
    # t1 = time.time()
    # for x in jnp.linspace(0.5, 1.0, 5001):
    #     _ = softening_function_jitted(x).block_until_ready()
    # print(_)
    # t2 = time.time()
    # print(t2 - t1)
    # print()
    #
    # print("- vmap+jit:")
    # t1 = time.time()
    # x = jnp.linspace(0.5, 1.0, 5001)
    # results = softening_function_vmapped(x)
    # print(results)
    # t2 = time.time()
    # print(t2 - t1)
