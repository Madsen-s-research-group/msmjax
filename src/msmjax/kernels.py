"""Functionality for splitting interaction kernels into sum of partial kernels.

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from msmjax.utils.general import CellMode, KernelFn, _divide_zero_safe, _sqrt


class SoftenerOneOverR:
    """Class for constructing and evaluating softener for the 1/r kernel.

    The softener is a function of a dimensionless argument rho that is equal
    to 1/rho for rho >= 1 and bounded and smooth for rho < 1.

    Args:
        order: Order (as a function of s = rho**2) of the Taylor polynomial
            that the softening function consists of for rho < 1.
            In line with common spline terminology, its polynomial degree (as
            a function of s = rho**2) is equal to order - 1.
    """

    # TODO: The `order` argument must be consistent with the `order` argument
    #  of the B-spline basis elements.

    def __init__(self, order: int):
        if not isinstance(order, (int, onp.integer, jnp.integer)):
            raise ValueError("'order' must be an integer.")
        if order < 1:
            raise ValueError("The expansion must at least be of order one.")
        # TODO: Should it be checked (at this point) whether the order is even?
        self.order = order
        last_coeff = 1.0
        coeffs = [last_coeff]
        for i_term in range(1, self.order):
            last_coeff *= (1.0 / i_term - 2) / 2.0
            coeffs.append(last_coeff)
        coeffs = list(reversed(coeffs))
        self.coeffs = jnp.array(coeffs)

    def __call__(self, rho):
        return jnp.where(
            rho < 1.0,
            jnp.polyval(self.coeffs, rho * rho - 1.0),
            _divide_zero_safe(1.0, rho),
        )


def split_one_over_r(
    max_level: int, level_zero_cutoff: float, softening_function: Callable
) -> list[KernelFn]:
    """Split kernel 1/r in (max_level + 1) terms according to reference.

    The splitting terms sum up to the Coulomb kernel 1/r like this:
    1/r = g_0(r) + g_1(r) + g_2(r) + ... + g_{max_order}(r)

    Args:
        max_level: The number of splits to be performed.
        level_zero_cutoff: The cutoff radius of the level-zero kernel function.
        softening_function: The basic smoothing function for this splitting.

    Returns:
        A list of one-argument functions g_l with l from zero to `max_order`
        that represent the terms in the splitting of the interaction kernel.
        These have their respective cutoffs 'built in' already (in the sense
        that they evaluate to zero for distances beyond) and take their
        arguments in the same length units that `level_zero_cutoff` was
        supplied in.

    Raises:
        ValueError: If the arguments do not make sense.
    """
    # TODO: Name of `max_level` parameter should be consistent with the rest
    #  of the code (also make sure that the correct variable name is used in
    #  docstring)
    if not isinstance(max_level, (int, onp.integer, jnp.integer)):
        raise ValueError("'max_level' must be an integer.")
    if max_level < 1:
        raise ValueError(
            "'max_level' must be at least one (which corresponds to the case "
            "of splitting the kernel in two terms)."
        )
    if level_zero_cutoff <= 0.0:
        raise ValueError("Cutoff must be positive.")

    def gamma_0(rho):
        return 1.0 / rho - softening_function(rho)

    def gamma_l(rho):
        return 2.0 * softening_function(2.0 * rho) - softening_function(rho)

    def gamma_L(rho):
        return 2.0 * softening_function(2.0 * rho)

    def g_l_factory(a_l, gamma):
        def g_l(r):
            nonlocal a_l, gamma
            return gamma(r / a_l) / a_l

        return g_l

    all_gammas = [gamma_0] + [gamma_l] * (max_level - 1) + [gamma_L]

    nruter = []
    a_l = level_zero_cutoff
    for ell, gamma in enumerate(all_gammas):
        nruter.append(g_l_factory(a_l, gamma))
        a_l *= 2.0

    return nruter


def _get_distances(extents_from_center, spacing_or_gridcell):
    indices_per_axis = [jnp.arange(-s, s + 1) for s in extents_from_center]
    indices = jnp.stack(
        jnp.meshgrid(*indices_per_axis, indexing="ij"), axis=-1
    )
    if jnp.ndim(spacing_or_gridcell) < 2:
        points = indices * spacing_or_gridcell
    else:
        points = indices @ spacing_or_gridcell
    return _sqrt((points * points).sum(axis=-1))


def _compute_one_stencil(
    function_values: ArrayLike, omega: ArrayLike, mode: str
):
    def _conv_1d(in1: ArrayLike, in2: ArrayLike):
        return jax.scipy.signal.convolve(in1, in2, mode=mode)

    result = function_values
    for axis in range(function_values.ndim):
        result = jnp.apply_along_axis(
            func1d=_conv_1d, axis=axis, arr=result, in2=omega
        )
    return result


def make_construct_stencils(
    omega: ArrayLike,
    n_levels_intermed: int,
    include_toplevel: bool,
    scaled_spacings: ArrayLike,
    cell_mode: CellMode,
    k_lowest_intermed: KernelFn = None,
    extents_from_center_intermed: tuple[int, ...] = None,
    k_toplevel: KernelFn = None,
    grid_shape_toplevel: tuple[int, ...] = None,
) -> Callable[[ArrayLike], list[Array]]:

    if n_levels_intermed < 0:
        raise ValueError("n_levels_intermed must be >= 0")
    if n_levels_intermed == 0 and not include_toplevel:
        raise ValueError(
            "n_levels_intermed = 0 and include_toplevel = False "
            "at the same is not allowed (this would mean that "
            "there isn't a single grid level)."
        )
    args_intermediate = [k_lowest_intermed, extents_from_center_intermed]
    if n_levels_intermed > 0 and any([x is None for x in args_intermediate]):
        raise ValueError(
            "k_lowest_intermed and extents_from_center_intermed "
            "are required when n_levels_intermed > 0."
        )
    args_toplevel = [k_toplevel, grid_shape_toplevel]
    if include_toplevel and any([x is None for x in args_toplevel]):
        raise ValueError(
            "k_toplevel and grid_shape_toplevel "
            "are required when include_toplevel = True."
        )

    def construct_stencils(cell: ArrayLike):
        if cell_mode == "ortho":
            spacings_or_gridcell_lowest = scaled_spacings * jnp.diag(cell)
        elif cell_mode == "triclinic":
            spacings_or_gridcell_lowest = (
                cell * scaled_spacings[:, jnp.newaxis]
            )
        else:
            raise ValueError("Invalid cell_mode")

        # Placeholder for level zero (l = 0), at which there is no grid:
        stencils = [None]

        # Intermediate levels (l = 1 ... L - 1):
        if n_levels_intermed > 0:
            distances_lvl_1 = _get_distances(
                extents_from_center_intermed, spacings_or_gridcell_lowest
            )
            kernel_values_at_gridpoints = k_lowest_intermed(distances_lvl_1)
            stencils.append(
                _compute_one_stencil(
                    kernel_values_at_gridpoints, omega, mode="same"
                )
            )
            for lvl in range(n_levels_intermed - 1):
                stencils.append(0.5 * stencils[-1])

        # Top level containing long-range tail (l = L), if included
        if include_toplevel:
            # TODO: Explain why the sizes are the way they are
            #  and why mode="valid".
            sizes_toplevel = tuple(
                (s - 1) + len(omega) // 2 for s in grid_shape_toplevel
            )
            max_grid_level = n_levels_intermed + 1
            distances_toplevel = _get_distances(
                sizes_toplevel,
                2 ** (max_grid_level - 1) * spacings_or_gridcell_lowest,
            )
            kernel_values_at_gridpoints = k_toplevel(distances_toplevel)
            stencils.append(
                _compute_one_stencil(
                    kernel_values_at_gridpoints, omega, mode="valid"
                )
            )

        return stencils

    return construct_stencils


def determine_min_kernel_stencil_size(
    cell: ArrayLike, spacings: ArrayLike, cutoff: float
):
    # TODO: Right module for this function? utils? bspline.gridops? core.longrange?
    #  (whichever it will be, currently it's probably not in the right one...)

    # TODO: Should the parameter names for spacings and r_cut suggest one
    #  specific grid level? In principle, if they're given at the same level,
    #  it does not matter which, since both are doubled at each level.
    #  But OTOH, the risk of inadvertently passing the level-ONE spacing and
    #  together with the level-ZERO cutoff should be minimized

    # TODO: unit test this function

    n_dim = cell.shape[0]
    inverse = onp.linalg.inv(cell)

    # TODO: Can this be made more generic (same code working for all dimensions)?
    # TODO: Could this be implemented via the usual max-cutoff formula
    #  (as implemented in `get_max_cutoff_3d`) instead? (Keep enlarging the
    #  cell passed to `get_max_cutoff_3d` in discrete steps, corresponding to
    #  adding an additional grid point, until the cutoff fits)
    if n_dim == 1:
        return tuple(onp.atleast_1d(cutoff / spacings).astype(int).tolist())
    elif n_dim == 2:
        phis = onp.linspace(0, 2 * onp.pi, 500)
        points_unitsphere = onp.array([onp.cos(phis), onp.sin(phis)]).T
    elif n_dim == 3:
        phis = onp.linspace(0, 2 * onp.pi, 200)
        thetas = onp.linspace(0, onp.pi, 200)
        phis, thetas = onp.meshgrid(phis, thetas)
        phis = phis.ravel()
        thetas = thetas.ravel()
        points_unitsphere = onp.array(
            [
                onp.cos(phis) * onp.sin(thetas),
                onp.sin(phis) * onp.sin(thetas),
                onp.cos(thetas),
            ]
        ).T
    else:
        raise ValueError("Spatial dimensions greater than 3 not supported.")

    points_at_cutoff = cutoff * points_unitsphere
    points_at_cutoff_transformed = points_at_cutoff @ inverse

    single_grid_cell = (
        cell
        / onp.linalg.norm(cell, axis=1)[:, onp.newaxis]
        * onp.atleast_1d(spacings)[:, onp.newaxis]
    )
    spacings_transformed = onp.diag(single_grid_cell @ inverse)

    # The maximum taken from the precomputed cutoff sphere points may be
    # slightly too low, because they incompletely sample the cutoff sphere
    # => use an additional small tolerance
    tol = 1.0e-3
    sizes_from_center = onp.floor(
        points_at_cutoff_transformed.max(axis=0) / spacings_transformed + tol
    ).astype(int)

    return tuple(sizes_from_center.tolist())
