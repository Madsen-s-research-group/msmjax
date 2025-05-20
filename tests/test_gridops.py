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

from msmjax.bspline.coefficients import compute_J_zeroplus

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


@pytest.fixture(params=[4, 6], scope="module")
def fixture_p(request) -> int:
    return request.param


side_lengths_1d = onp.array([5.0])
side_lengths_2d = onp.array([5.0, 7.5])
side_lengths_3d = onp.array([5.0, 7.5, 11.0])
pbc_1d = [(False,), (True,)]
pbc_2d = [(False, False), (True, True), (False, True)]
pbc_3d = [
    (False, False, False),
    (True, True, True),
    (False, True, False),
    (True, False, True),
]

box_params = (
    list(zip([side_lengths_1d] * len(pbc_1d), pbc_1d))
    + list(zip([side_lengths_2d] * len(pbc_2d), pbc_2d))
    + list(zip([side_lengths_3d] * len(pbc_3d), pbc_3d))
)


@pytest.fixture(params=box_params, scope="module")
def fixture_box(request):
    """Fixture combining box side lengths and periodic boundary conditions."""
    side_lengths, pbc = request.param
    return side_lengths, pbc


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
def test_gridcharges_restrict_vs_anterpolate_all_levels(
    fixture_particle_config, fixture_box, fixture_grid_params, fixture_p
):
    # TODO: Right place for this? Here or in tests for core.longrange?
    (pos, chg) = fixture_particle_config
    (shapes_all_levels, spacings_all_levels) = fixture_grid_params
    max_level = len(shapes_all_levels) - 1
    _, pbc = fixture_box
    p = fixture_p

    gridcharges_from_anterpolate = [None]
    for lvl in range(1, max_level + 1):
        grid_shape = shapes_all_levels[lvl]
        eval_basis = make_basis_evaluation_fn(
            grid_shape=grid_shape, p=p, pbc=pbc
        )
        # TODO: jit or not?
        basis_vals, basis_inds = jax.jit(eval_basis)(
            pos, spacings_all_levels[lvl]
        )
        # TODO: jit or not?
        gridcharges_from_anterpolate.append(
            _anterpolate(basis_vals, basis_inds, chg, grid_shape)
        )

    # TODO
    # for grid in fixture_grids[1:]:
    #     anterpolate = jax.jit(create_anterpolation_operator(grid))
    #     gridcharges_from_anterpolate.append(
    #         anterpolate(positions=pos, charges=chg)
    #     )

    for lvl in range(1, max_level + 1):
        restrict = make_restriction_operator(
            grid_shape_in=shapes_all_levels[lvl],
            grid_shape_out=shapes_all_levels[lvl + 1],
            p=p,
            pbc=pbc,
        )
        # TODO: jit or not?
        # TODO: tolerance? enable double precision?
        assert onp.allclose(
            jax.jit(restrict)(gridcharges_from_anterpolate[lvl]),
            gridcharges_from_anterpolate[lvl + 1],
        )


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
