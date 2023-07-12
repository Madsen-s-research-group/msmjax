#!/usr/bin/env python
"""Tests for the B-spline basis implementation"""
from contextlib import nullcontext

import jax
import jax.config
import numpy as np
import pytest
from scipy.interpolate import BSpline

from msmjax.bspline_basis import create_bspline_basis_element

jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=[3, 5, 7])
def fixture_order(request) -> int:
    return request.param


@pytest.fixture
def fixture_knots(fixture_order) -> np.ndarray:
    """Generate a set of knots for use with scipy.interpolate.BSpline

    Params:
        fixture_order: Order of the B-spline basis function

    Returns:
        Array containing the knot vector on the reference interval
        for specifying the B-spline basis element of given order
    """
    return np.arange(-(fixture_order + 1) / 2, (fixture_order + 3) / 2)


@pytest.fixture
def fixture_grid(fixture_order) -> np.ndarray:
    """Create a grid for testing the basis function evaluation

    Params:
        fixture_order: Order of the B-spline basis function

    Returns:
        Array containing a grid on the reference interval to evaluate the
        basis function on
    """
    return np.linspace(
        -(fixture_order + 1) / 2, (fixture_order + 1) / 2, num=51
    )


def test_initialize_too_low_order() -> None:
    """Test if initialization of basis element with order < 1 fails"""
    with pytest.raises(ValueError):
        create_bspline_basis_element(0)


def test_jit_compilation(fixture_grid) -> None:
    """Test if jit compilation does not fail"""
    basis = jax.jit(jax.vmap(create_bspline_basis_element()))
    with nullcontext():
        basis(fixture_grid)


def test_compare_value_to_scipy(fixture_order, fixture_knots, fixture_grid):
    """Test if implementation gives same values as scipy equivalent"""
    basis = jax.vmap(create_bspline_basis_element(fixture_order))
    scipy_basis = BSpline.basis_element(fixture_knots)
    assert np.allclose(basis(fixture_grid), scipy_basis(fixture_grid))


def test_compare_grad_to_scipy(fixture_order, fixture_knots, fixture_grid):
    """Test if implementation gradient gives same values as scipy equivalent"""
    grad = jax.vmap(jax.grad(create_bspline_basis_element(fixture_order)))
    scipy_basis = BSpline.basis_element(fixture_knots)
    assert np.allclose(grad(fixture_grid), scipy_basis(fixture_grid, nu=1))
