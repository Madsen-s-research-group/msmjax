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

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as onp


class SofteningFunctionOneOverR:
    # TODO: does it make sense for this to be a class?
    def __init__(self, order):
        if not isinstance(order, (int, onp.integer)):
            raise ValueError('"order" must be an integer')
        if order < 1:
            raise ValueError("the expansion must have at least one term")
        self.order = order
        last_coeff = 1.0
        coeffs = [last_coeff]
        for i_term in range(1, self.order):
            last_coeff *= (1.0 / i_term - 2) / 2.0
            coeffs.append(last_coeff)
        coeffs = list(reversed(coeffs))
        self.coeffs = jnp.array(coeffs)
        # TODO: does it make sense to implement derivative this way?
        # self.expansion = onp.polynomial.Polynomial(coeffs)
        # self.expansion_derivative = self.expansion.deriv(m=1)

    def __call__(self, rho):
        nruter = jnp.where(
            rho < 1.0, jnp.polyval(self.coeffs, rho * rho - 1.0), 1.0 / rho
        )
        return nruter


def build_splitting_one_over_r(
    max_level: int, level_zero_cutoff: float, softening_function: Callable
):
    """Split kernel 1/r in (max_order + 1) terms g_l(r) according to reference.

    The splitting terms sum up to the Coulomb kernel 1/r like this:
    1/r = g_0(r) + g_1(r) + g_2(r) + ... + g_{max_order}(r)

    Args:
        max_level: The number of splits to be performed.
        level_zero_cutoff: The cutoff radius of the level-zero kernel function
            (= of the direct term in the kernel splitting).
        softening_function: The basic smoothing function for this splitting.

    Returns:
        A list of one-argument functions g_l with l from zero to `max_order`
        that represent the summands in the splitting of the interaction
        kernel. These have their respective cutoff 'built in' already and
        their arguments must be given in the same length units that
        `level_zero_cutoff` was supplied in .

    Raises:
        ValueError: If the arguments do not make sense.
    """
    if not isinstance(max_level, (int, onp.integer, jnp.integer)):
        raise ValueError('"max_order" must be an integer')
    if max_level < 1:
        raise ValueError("the expansion must have at least one term")
    if level_zero_cutoff <= 0.0:
        raise ValueError('"cutoff" must be a positive number')
    # TODO
    # if not isinstance(softening_function, SmoothingFunction):
    #     raise ValueError("\"smoothing\" must be a SmoothingFunction instance")

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
