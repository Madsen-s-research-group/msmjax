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
