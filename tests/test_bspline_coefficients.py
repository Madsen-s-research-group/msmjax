"""Tests for the implementation of coefficients for B-spline interpolation

References:
    [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
    R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
    Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
    144 (11), 114112. https://doi.org/10.1063/1.4943868.
"""

import numpy as onp
import pytest

from msmjax.bspline.coefficients import compute_quasi_omega_prime

# Values from Table I in Ref. [1]:
omega_prime_ref_dict = {
    4: onp.array(
        [
            3.464,
            -1.732,
            0.679,
            -0.240,
            0.080,
            -0.026,
            0.008,
            -0.002,
            0.001,
            0.000,
            0.000,
            0.000,
            0.000,
        ]
    ),
    6: onp.array(
        [
            12.379,
            -9.377,
            5.809,
            -3.266,
            1.735,
            -0.889,
            0.444,
            -0.217,
            0.105,
            -0.050,
            0.024,
            -0.011,
            0.005,
        ]
    ),
    8: onp.array(
        [
            51.971,
            -45.671,
            34.575,
            -24.022,
            15.825,
            -10.061,
            6.237,
            -3.794,
            2.275,
            -1.348,
            0.792,
            -0.461,
            0.267,
        ]
    ),
    10: onp.array(
        [
            241.384,
            -225.114,
            189.064,
            -147.928,
            110.306,
            -79.525,
            55.945,
            -38.635,
            26.301,
            -17.700,
            11.801,
            -7.807,
            5.131,
        ]
    ),
    12: onp.array(
        [
            1190.122,
            -1140.060,
            1014.420,
            -853.182,
            688.291,
            -538.377,
            411.423,
            -308.820,
            228.557,
            -167.246,
            121.250,
            -87.226,
            62.340,
        ]
    ),
}


@pytest.fixture(params=[4, 6, 8, 10, 12], scope="module")
def fixture_p(request) -> int:
    """Fixture returning interpolation order"""
    return request.param


@pytest.fixture(scope="module")
def fixture_omega_prime_zeroplus(fixture_p):
    """Fixture returning reference values for omega'_m (m >= 0 part only)"""
    return omega_prime_ref_dict[fixture_p]


def test_compute_quasi_omega_prime_mu_inf(
    fixture_p, fixture_omega_prime_zeroplus
):
    """Test if computed quasi-interpolation omega_prime coefficients match
    the tabulated non-quasi (corresponding to mu = infinity) values for a
    large mu."""
    # Use some large mu
    mu = 30
    omega_calculated, _ = compute_quasi_omega_prime(p=fixture_p, mu=mu)
    omega_zeroplus_calculated = omega_calculated[len(omega_calculated) // 2 :]
    assert onp.allclose(
        onp.round(
            omega_zeroplus_calculated[: len(fixture_omega_prime_zeroplus)],
            3,
        ),
        fixture_omega_prime_zeroplus,
    )
