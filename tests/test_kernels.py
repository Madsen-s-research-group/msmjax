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

from msmjax.bspline_interpolation.coefficients import (
    compute_coeffs_with_truncation,
    compute_J_zeroplus,
)
from msmjax.gridops_multidim import set_up_grids_all_levels
from msmjax.kernels import (
    _construct_all_kernel_stencils,
    make_dynamic_kernel_stencil_construction_fn,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel
from msmjax.wrappers_old_code import (
    _construct_kernel_stencils as old_kernel_stencil_fn,
)

# TODO: Set this and no preallocate in a consistent way (either both via
#  environment variable, or both via jax.config.update)
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def fixture_level_zero_cutoff() -> float:
    return 2.5


@pytest.fixture(params=[2, 4, 6])
def fixture_softening_function(request) -> SofteningFunctionOneOverR:
    return SofteningFunctionOneOverR(order=request.param)


@pytest.fixture
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
    return split_one_over_r_kernel(
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
        SofteningFunctionOneOverR(order)


@pytest.mark.parametrize("order", [-1, 0])
def test_softening_function_err_order_too_low(order):
    """Test if initializing softening function fails for too low order."""
    with pytest.raises(ValueError, match=r".*one.*"):
        SofteningFunctionOneOverR(order)


def test_softening_function_continuity_at_one(
    fixture_softening_function, fixture_softening_function_derivatives
):
    """Test if softener and its derivatives are continuous with 1/rho at rho=1.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.
    """
    # Check the function itself
    target = 1.0
    assert jnp.isclose(fixture_softening_function(1.0), target)
    # ...and its derivatives.
    for k in range(1, fixture_softening_function.order):
        target *= -k
        assert jnp.isclose(
            fixture_softening_function_derivatives[k](1.0), target
        )


@pytest.mark.parametrize("fixture_softening_function", [2, 4], indirect=True)
def test_softening_function_derivatives_at_zero(
    fixture_softening_function, fixture_softening_function_derivatives
):
    """Test if the odd derivatives of the softener vanish at rho=0.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.

    Since this test involves derivatives of very high order, it is restricted
    to only lower-order softening functions for run time reasons.
    """
    for dgamma in fixture_softening_function_derivatives[1::2]:
        assert jnp.isclose(dgamma(0.0), 0.0)


@pytest.mark.parametrize("fixture_softening_function", [2, 4], indirect=True)
def test_softening_function_high_deriv_vanishes_globally(
    fixture_softening_function, fixture_softening_function_derivatives
):
    """Test if (2*order)-th derivative of softener vanishes for rho < 1.

    The condition being tested is a theoretical requirement on the softening
    function proposed in Ref. [1] and contained in Section II.A thereof.

    Since this test involves derivatives of very high order, it is restricted
    to only lower-order softening functions for run time reasons.
    """
    range_of_rho = jnp.linspace(0.0, 0.99, 100)
    assert jnp.allclose(
        jax.vmap(
            fixture_softening_function_derivatives[
                2 * fixture_softening_function.order
            ]
        )(range_of_rho),
        0.0,
    )


def test_split_err_max_level_noninteger(fixture_softening_function):
    """Test if kernel splitting fails for max level that is not integer."""
    with pytest.raises(ValueError):
        split_one_over_r_kernel(
            max_level=2.0,
            level_zero_cutoff=fixture_level_zero_cutoff,
            softening_function=fixture_softening_function,
        )


@pytest.mark.parametrize("max_level", [-1, 0])
def test_split_err_max_level_too_low(fixture_softening_function, max_level):
    """Test if kernel splitting fails for max level not at least one."""
    with pytest.raises(ValueError):
        split_one_over_r_kernel(
            max_level=max_level,
            level_zero_cutoff=fixture_level_zero_cutoff,
            softening_function=fixture_softening_function,
        )


@pytest.mark.parametrize("cutoff", [-2.5, 0.0])
def test_split_err_cutoff_not_positive(fixture_softening_function, cutoff):
    """Test if kernel splitting fails for cutoff that is zero or negative."""
    with pytest.raises(ValueError):
        split_one_over_r_kernel(
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


@pytest.mark.parametrize("p", [4, 6])
@pytest.mark.parametrize(
    "box_lengths",
    [
        onp.array([10.0]),
        onp.array([10.0, 12]),
        onp.array([10.0, 12.0, 15.0]),
    ],
)
def test_new_vs_old_stencil_construction_fn(
    fixture_partial_kernels, fixture_level_zero_cutoff, box_lengths, p
):
    if len(fixture_partial_kernels) == 2:
        pytest.xfail(
            "The kernel stencil fns incorrectly the case of a single grid level"
        )

    # TODO: Remove/replace this test in the long run. It is only meant to
    #  ensure we don't break anything while transitioning from the old (static,
    #  only one spacing value for all directions) to the new (dynamic,
    #  different spacings allowed) kernel stencil construction function.
    # TODO: Test with and without inclusion of top level -> in fact, the old
    #  function assumes that the top level is always included
    spacing_scalar = 1.0
    alpha = fixture_level_zero_cutoff / spacing_scalar
    mu = max(int(4 * alpha + p // 2), 3 * p // 2)
    n_levels = len(fixture_partial_kernels) - 1

    stencils_old = old_kernel_stencil_fn(
        kernels=fixture_partial_kernels,
        box_lengths=box_lengths,
        level_one_gridspacing=spacing_scalar,
        level_zero_cutoff=fixture_level_zero_cutoff,
        n_levels=n_levels,
        p=p,
        mu=mu,
    )

    omega, _ = compute_coeffs_with_truncation(p=p, mu=mu)
    cell = jnp.diag(box_lengths)
    spacings_one_per_axis = onp.full_like(box_lengths, spacing_scalar)
    sizes_from_center = onp.full_like(
        box_lengths, 2 * int(alpha) + 2, dtype=int
    )
    grids_new = set_up_grids_all_levels(
        box_lengths=box_lengths,
        level_one_spacings=spacings_one_per_axis,
        pbcs=(False,) * len(box_lengths),
        n_levels=n_levels,
        p=p,
        J_zeroplus=compute_J_zeroplus(p),
    )
    stencil_construction_fn_new = make_dynamic_kernel_stencil_construction_fn(
        kernel_fns=fixture_partial_kernels,
        sizes_from_center=sizes_from_center,
        reference_cell=cell,
        reference_spacings=spacings_one_per_axis,
        omega=omega,
        kernels_include_toplevel=True,  # because non-periodic
        sizes_from_center_toplevel=tuple(
            s + len(omega) // 2 for s in grids_new[-1].shape
        ),
    )
    stencils_new = stencil_construction_fn_new(cell)

    # Levels below top level
    for s_new, s_old in zip(stencils_new[1:-1], stencils_old[1:-1]):
        assert onp.allclose(s_new, s_old)

    # Top level: The new calculation does not automatically trim the top level
    # stencil to the grid size (in the future this should change), so we need
    # to do it manually for comparison.
    stencil_toplevel_old = stencils_old[-1]
    stencil_toplevel_new = stencils_new[-1]
    shape_diff = onp.array(stencil_toplevel_new.shape) - onp.array(
        stencil_toplevel_old.shape
    )
    excess_sizes = shape_diff // 2
    stencil_toplevel_new_trimmed = stencil_toplevel_new[
        tuple(slice(s, -s) for s in excess_sizes)
    ]
    assert onp.allclose(stencil_toplevel_new_trimmed, stencil_toplevel_old)
