"""Tests for grid operations specific to B-spline interpolation.

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

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from msmjax.bspline.gridops import (
    _arbitrary_dim_outer,
    _make_prolongate_1d,
    _make_restrict_1d,
    _multiindex_outer,
    _ravel_multi_index_with_invalidation,
    make_basis_evaluation_fn,
    make_prolongation_operator,
    make_restriction_operator,
)
from msmjax.core.longrange import _anterpolate


@pytest.mark.xfail(reason="Test not written yet")
def basis_particle_out_of_bounds():
    # TODO: test that the basis_eval_fn returns nan if a particle is located
    #  outside the cell along a non-periodic direction
    raise ValueError


@pytest.mark.xfail(reason="Test not written yet")
def test_anterpolate_explicit_loop():
    # TODO: Right place for this? Here or in tests for core.longrange?
    raise ValueError


@pytest.mark.xfail(reason="Test not written yet")
def test_gridcharges_sum_total():
    # TODO: Right place for this? Here or in tests for core.longrange?
    raise ValueError


@pytest.mark.xfail(reason="Test not written yet")
def test_gridcharges_restrict_vs_anterpolate_all_levels():
    # TODO: Right place for this? Here or in tests for core.longrange?
    raise ValueError


@pytest.mark.xfail(reason="Test not written yet")
def test_restrict_explicit_matmul():
    raise ValueError


@pytest.mark.xfail(reason="Test not written yet")
def test_prolongate_explicit_matmul():
    raise ValueError


@pytest.mark.xfail(reason="Test not written yet")
def test_particle_exactly_at_the_edge():
    # TODO: Test that there is no information loss when a particle is located
    #  at the very edge of the permissible range. But how to test? Calculate
    #  some (which?) quantity on a larger-than-necessary grid and check that
    #  the end result is unchanged?
    raise ValueError
