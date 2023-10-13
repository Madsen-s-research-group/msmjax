"""Tests for kernel splitting implementation.

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel


@pytest.fixture
def fixture_level_zero_cutoff():
    return 2.5


@pytest.fixture(params=[2, 4, 6])
def fixture_softening_function(request):
    return SofteningFunctionOneOverR(order=request.param)


@pytest.fixture(params=[1, 2, 3])
def fixture_partial_kernels(
    request, fixture_level_zero_cutoff, fixture_softening_function
):
    return split_one_over_r_kernel(
        max_level=request.param,
        level_zero_cutoff=fixture_level_zero_cutoff,
        softening_function=fixture_softening_function,
    )


def test_softening_function_continuity_at_one(fixture_softening_function):
    """Test if softener and its derivatives are continuous with 1/rho at rho=1.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.
    """
    # TODO: avoid duplication of setting up the derivative functions
    derivatives = [fixture_softening_function]
    for k in range(1, 2 * fixture_softening_function.order + 1):
        derivatives.append(jax.grad(derivatives[-1]))

    # Check the function itself
    target = 1.0
    assert jnp.isclose(derivatives[0](1.0), target)
    # ...and its derivatives.
    for k in range(1, fixture_softening_function.order):
        target *= -k
        assert jnp.isclose(derivatives[k](1.0), target)


@pytest.mark.parametrize("fixture_softening_function", [2, 4], indirect=True)
def test_softening_function_derivatives_at_zero(fixture_softening_function):
    """Test if the odd derivatives of the softener vanish at rho=0.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.

    Since this test involves derivatives of very high order, it is restricted
    to only lower-order softening functions for run time reasons.
    """
    # TODO: avoid duplication of setting up the derivative functions
    derivatives = [fixture_softening_function]
    for k in range(1, 2 * fixture_softening_function.order + 1):
        derivatives.append(jax.grad(derivatives[-1]))

    for dgamma in derivatives[1::2]:
        assert jnp.isclose(dgamma(0.0), 0.0)


@pytest.mark.parametrize("fixture_softening_function", [2, 4], indirect=True)
def test_softening_function_high_deriv_vanishes_globally(
    fixture_softening_function,
):
    """Test if (2*order)-th derivative of softening function vanishes globally.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.

    Since this test involves derivatives of very high order, it is restricted
    to only lower-order softening functions for run time reasons.
    """
    # TODO: avoid duplication of setting up the derivative functions
    derivatives = [fixture_softening_function]
    for k in range(1, 2 * fixture_softening_function.order + 1):
        derivatives.append(jax.grad(derivatives[-1]))

    range_of_rho = jnp.linspace(0.0, 0.99, 100)
    assert jnp.allclose(
        jax.vmap(derivatives[2 * fixture_softening_function.order])(
            range_of_rho
        ),
        0.0,
    )


def test_err_max_level_noninteger(fixture_softening_function):
    """Test if kernel splitting errors for max level that is not integer."""
    with pytest.raises(ValueError):
        split_one_over_r_kernel(
            max_level=2.0,
            level_zero_cutoff=fixture_level_zero_cutoff,
            softening_function=fixture_softening_function,
        )


@pytest.mark.parametrize("max_level", [-1, 0])
def test_err_max_level_not_at_least_one(fixture_softening_function, max_level):
    """Test if kernel splitting errors for max level not at least one."""
    with pytest.raises(ValueError):
        split_one_over_r_kernel(
            max_level=max_level,
            level_zero_cutoff=fixture_level_zero_cutoff,
            softening_function=fixture_softening_function,
        )


@pytest.mark.parametrize("cutoff", [-2.5, 0.0])
def test_err_cutoff_not_positive(fixture_softening_function, cutoff):
    """Test if kernel splitting errors for cutoff that is zero or negative."""
    with pytest.raises(ValueError):
        split_one_over_r_kernel(
            max_level=2,
            level_zero_cutoff=cutoff,
            softening_function=fixture_softening_function,
        )


def test_jit_compile_partial_kernels(
    fixture_partial_kernels, fixture_level_zero_cutoff
):
    """Test if jit compilation does not fail."""
    # TODO: avoid this code duplication
    max_cutoff = (
        2 ** (len(fixture_partial_kernels) - 1) * fixture_level_zero_cutoff
    )
    range_of_r = jnp.arange(
        0.01 * fixture_level_zero_cutoff,
        2 * max_cutoff,
        0.01,
    )
    for kernelfunc in fixture_partial_kernels:
        jitted_kernelfunc = jax.jit(jax.vmap(kernelfunc))
        jitted_kernelfunc(range_of_r)


def test_partial_kernels_sum_to_total(
    fixture_partial_kernels, fixture_level_zero_cutoff
):
    """Test if the partial kernels sum up to the total kernel."""
    # TODO: avoid this code duplication
    max_cutoff = (
        2 ** (len(fixture_partial_kernels) - 1) * fixture_level_zero_cutoff
    )
    range_of_r = jnp.arange(
        0.01 * fixture_level_zero_cutoff,
        2 * max_cutoff,
        0.01,
    )
    target = 1.0 / range_of_r
    summed = jnp.sum(
        jnp.vstack([k(range_of_r) for k in fixture_partial_kernels]), axis=0
    )

    assert jnp.allclose(summed, target)


def test_partial_kernels_cutoffs(
    fixture_partial_kernels, fixture_level_zero_cutoff
):
    """Test if different partial kernels go to zero at the correct distance."""
    for ell, k in enumerate(fixture_partial_kernels[:-1]):
        cutoff = 2**ell * fixture_level_zero_cutoff
        # The kernels have very small values already at distances
        # appreciably below the cutoff. To pass the greater-zero check,
        # we must therefore not get too close to the cutoff (at least for it
        # to work with JAX's default single precision arithmetic).
        r_below = jnp.arange(
            0.01 * fixture_level_zero_cutoff, 0.95 * cutoff, 0.01
        )
        assert (k(r_below) > 0.0).all()
        r_above = jnp.arange(cutoff, 2 * cutoff, 0.01)
        assert jnp.allclose(k(r_above), 0.0)
