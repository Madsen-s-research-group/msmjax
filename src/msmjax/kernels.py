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

from typing import Callable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax._src.basearray import ArrayLike

from msmjax.utils import _divide_zero_safe, _sqrt


class SofteningFunctionOneOverR:
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


def split_one_over_r_kernel(
    max_level: int, level_zero_cutoff: float, softening_function: Callable
):
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


def _compute_kernel_stencil(values: ArrayLike, omega: ArrayLike):
    def _onedim_convolution_fn(in1: ArrayLike, in2: ArrayLike):
        # TODO: method="fft"?
        return jax.scipy.signal.convolve(in1, in2, mode="same")

    convolved = values
    # TODO: exploit symmetry?
    # TODO: Is this sequential application of 1d convolutions the fastest thing
    #  one can do? Might it be faster to do it as a single 3D convolution
    #  (especially if the stencil size can be significantly reduced by symmetry?
    for i in range(values.ndim):
        convolved = jnp.apply_along_axis(
            func1d=_onedim_convolution_fn, axis=i, arr=convolved, in2=omega
        )
    return convolved


def _construct_all_kernel_stencils(
    kernel_fns: List[Callable],  # TODO: appropriate type hint?
    omega,
    points,  # TODO: pass points or directly the distances?
    kernels_include_toplevel: bool,
    points_toplevel,  # TODO: pass points or directly the distances?
):
    # TODO: raise error when `kernels_include_toplevel=True`, but sizes not given
    highest_included_level = len(kernel_fns) - 1

    # Level zero (at which there is no grid)
    stencils = [None]

    # TODO: If there is only one level (i.e. kernel_fns has length 2, with
    #  `None` as the first element), the top-level treatment in this
    #  function is wrong! Level one is always computed, and the
    #  `if kernels_include_toplevel: ... else:` always adds another level!
    #  The old (static) kernel stencil construction fn suffers from the same
    #  problem.

    # Level one
    distances = _sqrt((points * points).sum(axis=-1))
    fn_vals_at_points = kernel_fns[1](distances)
    stencils.append(_compute_kernel_stencil(fn_vals_at_points, omega))

    # Intermediate levels:
    # For the type of kernel splitting used here, the intermediate-level
    # kernel values (and thus stencils) can be computed by simply dividing
    # the one from the previous level by 2. But this need not hold for
    # other kernels or ways of splitting!
    # TODO: Can this be done faster by a broadcast multiplication? So far, it
    #  looks like there is not much to be gained here. The stencil calculation
    #  appears to be not much of a bottleneck.
    for lvl in range(2, highest_included_level):
        stencils.append(0.5 * stencils[-1])

    # Highest included level
    if kernels_include_toplevel:
        # TODO: Some possible efficiency gain by precomputing `points_cartesian`
        #  or `points_cartesian_toplevel`, whichever is larger in shape,
        #  and then getting the smaller by indexing into the larger
        # TODO: For the size of the top level stencil chosen sufficiently
        #  large (I think it needs to be the grid size + half the length of
        #  omega as padding), constructing it is very costly
        # TODO: The use of `linalg.norm` instead of custom `_sqrt` might lead
        #  to problems.
        # TODO: highest_included_level -1 or -2? (I think -2 might be a remnant
        #  from when highest_included_level was defined differently)
        #  ...but should this scaling be done inside this function at all,
        #  maybe it should receive the correct distances from outside?
        distances_toplevel = 2 ** (highest_included_level - 1) * _sqrt(
            (points_toplevel * points_toplevel).sum(axis=-1)
        )
        fn_vals_at_points_toplevel = kernel_fns[-1](distances_toplevel)
        stencils.append(
            _compute_kernel_stencil(fn_vals_at_points_toplevel, omega)
        )
    else:
        stencils.append(0.5 * stencils[-1])

    return stencils


def make_dynamic_kernel_stencil_construction_fn(
    kernel_fns: List[Callable],
    sizes_from_center: Sequence[int],
    reference_cell,
    reference_spacings,
    omega,
    kernels_include_toplevel: bool,
    sizes_from_center_toplevel=None,
):
    # TODO: "dynamic" in name?
    # TODO: raise error when `kernels_include_toplevel=True`, but sizes not given

    reference_side_lengths = onp.linalg.norm(reference_cell, axis=1)
    reference_spacings = onp.asarray(reference_spacings)
    spacings_unitcube = reference_spacings / reference_side_lengths

    indices_1d = [onp.arange(-s, s + 1) for s in sizes_from_center]
    indices = onp.stack(onp.meshgrid(*indices_1d, indexing="ij"), axis=-1)
    points_unitcube = indices * spacings_unitcube
    if kernels_include_toplevel:
        indices_1d_toplevel = [
            onp.arange(-s, s + 1) for s in sizes_from_center_toplevel
        ]
        indices_toplevel = onp.stack(
            onp.meshgrid(*indices_1d_toplevel, indexing="ij"), axis=-1
        )
        points_unitcube_toplevel = indices_toplevel * spacings_unitcube

    def construct_kernel_stencils(cell):
        return _construct_all_kernel_stencils(
            kernel_fns=kernel_fns,
            omega=omega,
            points=points_unitcube @ cell,
            kernels_include_toplevel=kernels_include_toplevel,
            points_toplevel=(
                points_unitcube_toplevel @ cell
                if kernels_include_toplevel
                else None
            ),
        )

    return construct_kernel_stencils
