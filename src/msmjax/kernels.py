"""Code for splitting interaction kernels into sum of partial kernels.

References:
    [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
    R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
    Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
    144 (11), 114112. https://doi.org/10.1063/1.4943868.

    [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
    Forces for the Simulation of Biomolecules (PhD thesis), University
    of Illinois at Urbana-Champaign, 2006.
"""

from typing import Callable, Sequence

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

    def __init__(self, order: int):
        if not isinstance(order, (int, onp.integer, jnp.integer)):
            raise ValueError("'order' must be an integer.")
        if order < 1:
            raise ValueError("The expansion must at least be of order one.")
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
        arguments in the same length units that ``level_zero_cutoff`` was
        supplied in.

    Raises:
        ValueError: If the arguments do not make sense.
    """
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


def _get_distances(
    extents_from_center: Sequence[int], spacing_or_gridcell: ArrayLike
) -> Array:
    """Compute distances (from the origin) of points on a regular grid.

    This function's originally intended use is to construct the grid of
    distances at which the interaction kernels are to be evaluated for the
    purpose of calculating kernel stencils for interpolation.

    Args:
        extents_from_center: Number of points along each axis, counted from,
            and not including, the center point. That is to say, for
            ``extents_from_center = (s_1, s_2, s_3)``, the shape of the grid
            of points is ``(2 * s_1 + 1, 2 * s_2 + 1, 2 * s_3 + 1)``.
        spacing_or_gridcell: Either a scalar grid spacing value (use if
            orthogonal axes with the same grid spacing along all axes),
            a 1-d array (use if orthogonal axes with different spacings),
            or a 2-d array of shape ``(n_dim, n_dim)`` corresponding to
            exactly one grid cell (use if the axes spanning the grid are not
            orthogonal).

    Returns:
        Distances from the origin of points on the grid.
    """
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
    function_values: ArrayLike, omega_prime: ArrayLike, mode: str
) -> Array:
    """Compute one kernel stencil by convolution with universal spline coeffs.

    Args:
        function_values: Values of the function for which the stencil of
            interpolation coefficients is to be computed, on a regular grid.
        omega_prime: 1-d sequence of universal spline coefficients for
            convolution with ``function_values``.
        mode: The ``mode`` argument passed to ``jax.scipy.signal.convolve``
            used internally for doing the convolution.

    Returns:
        The resulting kernel stencil.
    """

    def _conv_1d(in1: ArrayLike, in2: ArrayLike):
        return jax.scipy.signal.convolve(in1, in2, mode=mode)

    result = function_values
    for axis in range(function_values.ndim):
        result = jnp.apply_along_axis(
            func1d=_conv_1d, axis=axis, arr=result, in2=omega_prime
        )
    return result


def make_construct_stencils(
    omega_prime: ArrayLike,
    n_levels_intermed: int,
    include_toplevel: bool,
    scaled_spacings: ArrayLike,
    cell_mode: CellMode,
    k_lowest_intermed: KernelFn | None = None,
    extents_from_center_intermed: tuple[int, ...] | None = None,
    k_toplevel: KernelFn | None = None,
    grid_shape_toplevel: tuple[int, ...] | None = None,
) -> Callable[[ArrayLike], list[Array]]:
    """Create a function to compute all kernel stencils from a given unit cell

    The created function is suitable for passing as the
    ``kernel_stencil_construction_fn`` argument to
    :py:func:`msmjax.core.longrange.make_compute_u_oneplus`.

    The design is taileored to kernel splitting schemes like the one
    implemented by :func:`split_one_over_r`, in which all the
    intermediate-level partial kernels :math:`k_l(r)`, :math:`l = 1, ...,
    L - 1` have the same functional form except for scaling, and the kernel
    stencils on grids at subsequent levels with twice the spacing can be
    computed by simply dividing the lower-level kernel stencil by two.

    Args:
        omega_prime: 1-d sequence of universal spline coefficients from
            convolution with which the kernel stencils are obtained.
        n_levels_intermed: The number of intermediate levels, i.e. all
            levels :math:`l = 1, ..., L-1`, but excluding :math:`l = 0` and
            :math:`l = L`.
        include_toplevel: Whether the top level (:math:`l = L`) should be
            included. Usually, this is true in systems without periodicity,
            and false in periodic ones.
        scaled_spacings: 1-d array of grid spacings values, one per axis,
            expressed in fractional coordinates w.r.t. the cell.
        cell_mode: A string specifying assumptions on the shape of the
            unit cell. Either the cell is assumed orthorhombic and
            axis-aligned, in which case only its diagonal is considered,
            reducing computational cost, or a general triclinic one.
        k_lowest_intermed: The partial kernel at the lowest intermediate grid
            level (:math:`l = 1 < L` if ``n_levels_intermed`` > 0).
            Omit if ``n_levels_intermed = 0``, in which case :math:`L = 1`.
        extents_from_center_intermed: Number of points of the intermediate
            (:math:`l = L`) stencils along each axis, counted from, and not
            including, the center point. That is to say, for
            ``extents_from_center = (s_1, s_2, s_3)``, the shape of the
            stencils is ``(2 * s_1 + 1, 2 * s_2 + 1, 2 * s_3 + 1)``.
            Required if ``k_toplevel`` was given.
        k_toplevel: The partial kernel :math`k_L(r)` at the top level.
        grid_shape_toplevel: The shape (number of points along each axis) of
            the highest-level (:math:`l = L`) grid. Required if ``k_toplevel``
            was given.
    Returns:
        A function of one argument (cell, of shape ``(n_dim, n_ndim)``), that
        returns a list of kernel stencils, one for each level, with ``None``
        as a placeholder at level zero, where there is no grid.
    """
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
                    kernel_values_at_gridpoints, omega_prime, mode="same"
                )
            )
            for lvl in range(n_levels_intermed - 1):
                stencils.append(0.5 * stencils[-1])

        # Top level containing long-range tail (l = L), if included.
        # Since there is no cutoff, and the size of the stencil is always
        # determined such that it covers the entire grid. This only really
        # makes sense if there is no periodicity.
        if include_toplevel:
            sizes_toplevel = tuple(
                (s - 1) + len(omega_prime) // 2 for s in grid_shape_toplevel
            )
            max_grid_level = n_levels_intermed + 1
            distances_toplevel = _get_distances(
                sizes_toplevel,
                2 ** (max_grid_level - 1) * spacings_or_gridcell_lowest,
            )
            kernel_values_at_gridpoints = k_toplevel(distances_toplevel)
            stencils.append(
                _compute_one_stencil(
                    kernel_values_at_gridpoints, omega_prime, mode="valid"
                )
            )

        return stencils

    return construct_stencils
