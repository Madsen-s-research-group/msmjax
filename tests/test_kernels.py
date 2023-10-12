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
