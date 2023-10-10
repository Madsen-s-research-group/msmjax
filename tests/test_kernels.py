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
