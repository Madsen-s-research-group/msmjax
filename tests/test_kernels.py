import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import numpy as onp
import pytest

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel


@pytest.fixture
def level_zero_cutoff():
    return 2.5


@pytest.fixture
def partial_kernels(level_zero_cutoff):
    return split_one_over_r_kernel(
        max_level=3,  # TODO: parametrize?
        level_zero_cutoff=level_zero_cutoff,
        softening_function=SofteningFunctionOneOverR(
            order=4
        ),  # TODO: (how to) make into fixture? Should order be parametrized?
    )


def test_partial_kernels_summing_up_to_total(
    partial_kernels, level_zero_cutoff
):
    """Test if the partial kernels sum up to the total kernel."""
    max_cutoff = 2 ** (len(partial_kernels) - 1) * level_zero_cutoff
    range_of_r = jnp.arange(
        0.01 * level_zero_cutoff,
        2 * max_cutoff,
        0.01,
    )
    target = 1.0 / range_of_r
    summed = jnp.sum(
        jnp.vstack([k(range_of_r) for k in partial_kernels]), axis=0
    )

    assert jnp.allclose(summed, target)


def test_partial_kernels_cutoffs(partial_kernels, level_zero_cutoff):
    """Test if different partial kernels go to zero at the correct distance."""
    for ell, k in enumerate(partial_kernels[:-1]):
        cutoff = 2**ell * level_zero_cutoff
        # The kernels have very small values already at distances
        # appreciably below the cutoff. To pass the greater-zero check,
        # we must therefore not get too close to the cutoff (at least for it
        # to work with JAX's default single precision arithmetic).
        r_below = jnp.arange(0.01 * level_zero_cutoff, 0.95 * cutoff, 0.01)
        assert (k(r_below) > 0.0).all()
        r_above = jnp.arange(cutoff, 2 * cutoff, 0.01)
        assert jnp.allclose(k(r_above), 0.0)
