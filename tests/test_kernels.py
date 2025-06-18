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

from typing import Callable, List

import jax
import jax.numpy as jnp
import pytest
from jax import config

from msmjax.kernels import SoftenerOneOverR, split_one_over_r

# TODO: Set this and no preallocate in a consistent way (either both via
#  environment variable, or both via config.update)
config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def fixture_level_zero_cutoff() -> float:
    return 2.5


@pytest.fixture(params=[2, 4], scope="module")
def fixture_p(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def fixture_softening_function(fixture_p) -> SoftenerOneOverR:
    return SoftenerOneOverR(order=fixture_p)


@pytest.fixture(scope="module")
def fixture_softening_function_derivatives(
    fixture_softening_function,
) -> List[Callable]:
    derivatives = [fixture_softening_function]
    for k in range(1, 2 * fixture_softening_function.order + 1):
        derivatives.append(jax.grad(derivatives[-1]))

    return derivatives


@pytest.fixture(params=[1, 2, 3])
def fixture_partial_kernels(
    request, fixture_level_zero_cutoff, fixture_softening_function
) -> List[Callable]:
    return split_one_over_r(
        max_level=request.param,
        level_zero_cutoff=fixture_level_zero_cutoff,
        softening_function=fixture_softening_function,
    )


@pytest.fixture()
def fixture_range_of_r(fixture_level_zero_cutoff, fixture_partial_kernels):
    """Generate a range of distance values covering all relevant cutoffs."""
    max_cutoff = (
        2 ** (len(fixture_partial_kernels) - 1) * fixture_level_zero_cutoff
    )
    return jnp.arange(
        0.01 * fixture_level_zero_cutoff,
        2 * max_cutoff,
        0.01,
    )


@pytest.mark.parametrize("order", [1.0, 1.5])
def test_softening_function_err_order_noninteger(order):
    """Test if initializing softening function fails for noninteger order."""
    with pytest.raises(ValueError, match=r".*integer.*"):
        SoftenerOneOverR(order)


@pytest.mark.parametrize("order", [-1, 0])
def test_softening_function_err_order_too_low(order):
    """Test if initializing softening function fails for too low order."""
    with pytest.raises(ValueError, match=r".*one.*"):
        SoftenerOneOverR(order)


def test_softening_function_continuity_at_one(
    fixture_softening_function, fixture_softening_function_derivatives
):
    """Test if softener and its derivatives are continuous with 1/rho at rho=1.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.
    """
    # Check the function itself
    target = 1.0
    assert jnp.isclose(jax.jit(fixture_softening_function)(1.0), target)
    # ...and its derivatives.
    for k in range(1, fixture_softening_function.order):
        target *= -k
        assert jnp.isclose(
            jax.jit(fixture_softening_function_derivatives[k])(1.0), target
        )


def test_softening_function_derivatives_at_zero(
    fixture_softening_function_derivatives,
):
    """Test if the odd derivatives of the softener vanish at rho=0.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.

    For high orders p, this takes a long time.
    """
    for dgamma in fixture_softening_function_derivatives[1::2]:
        assert jnp.isclose(jax.jit(dgamma)(0.0), 0.0)


def test_softening_function_high_deriv_vanishes_globally(
    fixture_p,
    fixture_softening_function_derivatives,
):
    """Test if (2*order)-th derivative of softener vanishes for rho < 1.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.

    For high orders p, this takes a long time.
    """
    range_of_rho = jnp.linspace(0.0, 0.99, 100)
    assert jnp.allclose(
        jax.jit(
            jax.vmap(fixture_softening_function_derivatives[2 * fixture_p])
        )(range_of_rho),
        0.0,
    )


def test_split_err_max_level_noninteger(fixture_softening_function):
    """Test if kernel splitting fails for max level that is not integer."""
    with pytest.raises(ValueError):
        split_one_over_r(
            max_level=2.0,
            level_zero_cutoff=fixture_level_zero_cutoff,
            softening_function=fixture_softening_function,
        )


@pytest.mark.parametrize("max_level", [-1, 0])
def test_split_err_max_level_too_low(fixture_softening_function, max_level):
    """Test if kernel splitting fails for max level not at least one."""
    with pytest.raises(ValueError):
        split_one_over_r(
            max_level=max_level,
            level_zero_cutoff=fixture_level_zero_cutoff,
            softening_function=fixture_softening_function,
        )


@pytest.mark.parametrize("cutoff", [-2.5, 0.0])
def test_split_err_cutoff_not_positive(fixture_softening_function, cutoff):
    """Test if kernel splitting fails for cutoff that is zero or negative."""
    with pytest.raises(ValueError):
        split_one_over_r(
            max_level=2,
            level_zero_cutoff=cutoff,
            softening_function=fixture_softening_function,
        )


def test_partial_kernels_jit_compilation(
    fixture_partial_kernels, fixture_range_of_r
):
    """Test if jit compilation does not fail."""
    for kernelfunc in fixture_partial_kernels:
        jitted_kernelfunc = jax.jit(jax.vmap(kernelfunc))
        jitted_kernelfunc(fixture_range_of_r)


def test_partial_kernels_sum_to_total(
    fixture_partial_kernels, fixture_range_of_r
):
    """Test if the partial kernels sum up to the total kernel."""
    target = 1.0 / fixture_range_of_r
    summed = jnp.sum(
        jnp.vstack(
            [kernel(fixture_range_of_r) for kernel in fixture_partial_kernels]
        ),
        axis=0,
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
