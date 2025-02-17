"""Generic code for long-range part (that is evaluated using grids).

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

from functools import partial
from typing import Callable, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

# TODO: Define in some global typedef or utils module?
CellMode = Literal["ortho", "general"]
BasisEvalFn = Callable[[ArrayLike], tuple[Array, Array]]
BasisGradFn = Callable[[ArrayLike], tuple[Array, Array]]
BasisValAndGradFn = Callable[[ArrayLike], tuple[tuple[Array, Array], Array]]


@partial(jax.jit, static_argnames=["pbc", "method"])
def special_periodic_convolve(
    data: ArrayLike,
    kernel: ArrayLike,
    pbc: Sequence[bool],
    method: Literal["direct", "fft"],  # TODO: centralize definition? default?
) -> Array:
    """Perform a specialized case of convolution with optional wrapping.

    TODO: Show the formula of what this is meant for (convolving grid charge
        with interaction kernel coefficient stencil)?

    Implemented as a wrapper around :func:`jax.scipy.signal.convolve`
    with, depending on periodicity, appropriate padding of the input arrays:

        - If no direction is periodic, this function is equivalent to calling
          :func:`jax.scipy.signal.convolve` with `mode='same'`.
        - Along any periodic direction, the ``data`` array is first
          periodically replicated as much as needed for the ``kernel`` array
          to not extend beyond the edges. Then, the convolution is performed
          with :func:`jax.scipy.signal.convolve`, before trimming the result
          back to the original size of ``data``.

    Args:
        data: First input. Represents some data on a real-space grid.
            If a direction is periodic, it corresponds to the values contained
            within the unit cell along that direction.
        kernel: Second input. Should have the same number of dimensions as
            ``data``. Represents a finite-size kernel or filter. In the
            original intended use case, always has an odd number of points
            along each dimension (i.e., can be centered w.r.t. the points of
            ``data``).
        pbc: One boolean per direction signaling periodicity.
        method: String indicating the method to use for calculating the
            convolution. Either 'direct' or 'fft'. Passed on to
            :func:`jax.scipy.signal.convolve`.

    Returns:
        An array of the same shape as ``data`` containing the convolution of
        the two arrays.
    """
    pbc = onp.asarray(pbc)

    if pbc.any():
        size_kernel = onp.array(kernel.shape)
        size_kernel_below_middle = size_kernel // 2
        size_kernel_above_middle = size_kernel - size_kernel_below_middle - 1
        pad_width = tuple(
            (int(s_b), int(s_a))
            for s_b, s_a in zip(
                size_kernel_below_middle, size_kernel_above_middle
            )
        )
        pad_width = tuple(
            pw if periodic else (0, 0) for pw, periodic in zip(pad_width, pbc)
        )
        inds_reconstruct_unpadded = []
        for pw, periodic in zip(pad_width, pbc):
            if periodic:
                inds_reconstruct_unpadded.append(slice(pw[0], -pw[1]))
            else:
                inds_reconstruct_unpadded.append(slice(None))
        inds_reconstruct_unpadded = tuple(inds_reconstruct_unpadded)
        data_extended = jnp.pad(
            data,
            pad_width=pad_width,
            mode="wrap",
        )
        nruter = jax.scipy.signal.convolve(
            data_extended, kernel, mode="same", method=method
        )
        return nruter[inds_reconstruct_unpadded]
    else:
        return jax.scipy.signal.convolve(
            data, kernel, mode="same", method=method
        )


def _anterpolate(
    basis_vals: ArrayLike,
    indices: ArrayLike,
    charges: ArrayLike,
    grid_shape: tuple[int, ...],
):
    grid_size = int(onp.prod(grid_shape))
    gridcharge_flat = jnp.zeros(grid_size)
    gridcharge_flat = gridcharge_flat.at[indices].add(
        charges[:, jnp.newaxis] * basis_vals
    )
    return gridcharge_flat.reshape(grid_shape)


def make_anterpolation_fn(
    basis_eval_fn: BasisEvalFn,
    grid_shape: tuple[int, ...],
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """

    Args:
        basis_eval_fn: A function that, for all particles, identifies those
            grid points with non-zero basis function values, and evaluates
            them. See :func:`make_energy_interpolation_fn` for details of
            signature and meaning of return values.
        grid_shape: Tuple of integers indicating the shape of the target grid
            to which to anterpolate the particle charges.

    Returns:
        A function that takes arrays of particle positions and charges and
        returns array of grid charge.
    """
    grid_size = int(onp.prod(grid_shape))

    def anterpolate(positions: ArrayLike, charges: ArrayLike) -> Array:
        """Anterpolate charge from particles to grid"""
        # TODO: Indicate by variable names that indices are expected to be flat?
        # TODO: Behavior when basis_eval_fn returns out-of-bounds indices?
        #  Are there reasonable cases in which this may occur? Warning in docstring?
        basis_vals, indices = basis_eval_fn(positions)
        gridcharge_flat = jnp.zeros(grid_size)
        gridcharge_flat = gridcharge_flat.at[indices].add(
            charges[:, jnp.newaxis] * basis_vals
        )
        return gridcharge_flat.reshape(grid_shape)

    return anterpolate


def _interpolate_potential():
    # TODO: Do we want to provide this?
    pass


def _interpolate_energy(
    gridpotential: ArrayLike,
    basis_vals: ArrayLike,
    indices: ArrayLike,
    charges: ArrayLike,
) -> Array:
    """Low-level function for calculating energy from grid potential.

    Args:
        gridpotential: Array of grid potential (:math:`e^{l+}` in the language
            of the reference).
        basis_vals: TODO: How best to document (appears in several places)? More informative variable name? (`per_particle_basis_vals`?)
        indices: TODO: How best to document (appears in several places)? More informative variable name? (`per_particle_inds`?)
        charges: Array of particle charges, shape `(n_particles,)`.

    Returns:
        Scalar electrostatic energy.
    """
    # TODO: Do we need to use a fill value with `take` here?
    #  (it shouldn't be possible for indices returned by the spline eval
    #  functions to be out of bounds)
    energy = 0.5 * jnp.sum(
        charges * (gridpotential.take(indices) * basis_vals).sum(axis=1)
    )
    return energy


def _interpolate_forces(
    gridpotential: ArrayLike,
    basis_grads: ArrayLike,
    indices: ArrayLike,
    charges: ArrayLike,
) -> Array:
    """Low-level function for calculating forces from grid potential.

    Args:
        gridpotential: Array of grid potential (:math:`e^{l+}` in the language
            of the reference).
        basis_grads: TODO: How best to document (appears in several places)? More informative variable name? (`per_particle_basis_grads`?)
        indices: TODO: How best to document (appears in several places)? More informative variable name? (`per_particle_inds`?)
        charges: Array of particle charges, shape `(n_particles,)`.

    Returns:
        Array of forces on particles, shape `(n_particles, n_dim)`.
    """
    # TODO: Do we need to use a fill value with `take` here?
    #  (it shouldn't be possible for indices returned by the spline eval
    #  functions to be out of bounds)
    forces = -charges[:, jnp.newaxis] * jnp.sum(
        gridpotential.take(indices)[..., jnp.newaxis] * basis_grads, axis=1
    )
    return forces


def make_energy_interpolation_fn(
    basis_eval_fn: BasisEvalFn,  # TODO: name? "per_particle"? (also change in docstrings everywhere)
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], Array]:
    """Create a function that computes the energy by interpolating potential.

    Args:
        basis_eval_fn: A function that, for each particle,

                1) identifies all grid points that are sufficiently close for the
                   particle's position to be contained within the support of
                   the associated basis functions, i.e., finds the set of
                   grid points
                   :math:`M = \\{
                   \\mathbf{m} : \\varphi_{\\mathbf{m}}(\\mathbf{r}_i) \\neq 0
                   \\} \\,`,
                   (where :math:`\\mathbf{r}_i` denotes the position of
                   particle :math:`i`),

                2) evaluates the corresponding basis functions, i.e.
                   computes :math:`\\varphi_{\\mathbf{m}}(\\mathbf{r}_i)`
                   for all :math:`\mathbf{m} \in M \\,`.

            Inputs and outputs:

                - Input to ``basis_eval_fn`` should be a 2-d array of
                  particle positions, shape `(n_particles, n_dim)`.

                - Output of ``basis_eval_fn`` should be a tuple of two
                  2-d arrays of shape `(n_particles, support_size)`,
                  where `support_size` designates the fixed number of
                  non-zero basis functions around each particle (= the
                  cardinality of :math:`M` from above).
                  Their first axes run over particles, and the second over
                  grid points.
                  The first of the two arrays contains the values of the basis
                  functions for each particle, and the second array contains
                  the `flat` (!) indices of the corresponding grid points.

    Returns:
        A function of three array arguments (grid potential, particle
        positions, particle charges) that calculates the energy.
    """

    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ) -> Array:
        basis_vals, indices = basis_eval_fn(positions)
        return _interpolate_energy(gridpotential, basis_vals, indices, charges)

    return compute


def make_forces_interpolation_fn(
    basis_grad_fn: BasisGradFn,  # TODO: name? "per_particle"? (also change in docstrings)
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], Array]:
    """Create a function that computes the forces by interpolating potential.

    Args:
        basis_grad_fn: A function like the ``basis_eval_fn`` passed to
            :func:`make_energy_interpolation_fn` (see there for details),
            except it evaluates the gradients of the basis functions w.r.t.
            to particle positions,
            :math:`\\nabla_{\\mathbf{r}_i} \\,
            \\varphi_{\\mathbf{m}}(\\mathbf{r}_i), \\,
            \mathbf{m} \\in M`.

            Consequently, the shape of the two output arrays, for the values
            of the gradient and the indices of the non-zero grid points
            around each particle, should be
            `(n_particles, support_size, n_dim)` and
            `(n_particles, support_size)`.

    Returns:
        A function of three array arguments (grid potential, particle
        positions, particle charges) that calculates the forces.
    """

    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ) -> Array:
        basis_grads, indices = basis_grad_fn(positions)
        return _interpolate_forces(
            gridpotential, basis_grads, indices, charges
        )

    return compute


def make_energy_and_forces_interpolation_fn(
    basis_val_and_grad_fn: BasisValAndGradFn,
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], tuple[Array, Array]]:
    """Create function that computes energy, forces by interpolating potential.

    Args:
        basis_val_and_grad_fn: TODO: How to document? Unlike for `basis_eval_fn`, there is no other place where this is used, that could be referenced.

    Returns:
        A function of three array arguments (grid potential, particle
        positions, particle charges) that calculates the energy and forces.
    """

    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ) -> tuple[Array, Array]:
        # TODO: Make sure that the expected structure of the return value of
        #  basis_val_and_grad_fn is properly documented, and correctly used
        #  elsewhere.
        (basis_vals, indices), basis_grads = basis_val_and_grad_fn(positions)
        energy = _interpolate_energy(
            gridpotential, basis_vals, indices, charges
        )
        forces = _interpolate_forces(
            gridpotential, basis_grads, indices, charges
        )
        return energy, forces

    return compute


def make_unitcube_transform_fns_ortho(
    cell: ArrayLike,
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array]]:
    # TODO: Remove if superseded by _make_unitcube_transform_fns
    # TODO: Right module for this function?
    inverse = 1.0 / jnp.diag(cell)
    transform_pos = lambda x: x * inverse
    backtransform_grad = lambda dx: dx * inverse
    return transform_pos, backtransform_grad


def make_unitcube_transform_fns_general(
    cell: ArrayLike,
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array]]:
    # TODO: Remove if superseded by _make_unitcube_transform_fns
    # TODO: Right module for this function?
    inverse = jnp.linalg.pinv(cell)
    transform_pos = lambda x: x @ inverse
    backtransform_grad = lambda dx: dx @ inverse.T
    return transform_pos, backtransform_grad


def _make_unitcube_transform_fns(
    cell: ArrayLike | None, transform_mode: CellMode | None
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array]]:
    if transform_mode is None:
        transform_pos = lambda x: x
        backtransform_grad = lambda x: x
        return transform_pos, backtransform_grad
    elif transform_mode == "ortho":
        inverse = 1.0 / jnp.diag(cell)
        transform_pos = lambda x: x * inverse
        backtransform_grad = lambda dx: dx * inverse
        return transform_pos, backtransform_grad
    elif transform_mode == "general":
        inverse = jnp.linalg.pinv(cell)
        transform_pos = lambda x: x @ inverse
        backtransform_grad = lambda dx: dx @ inverse.T
        return transform_pos, backtransform_grad
    else:
        raise ValueError(f"Invalid mode: {transform_mode}")


def make_grid_pass(
    restriction_fns: Sequence[Callable],
    prolongation_fns: Sequence[Callable],
    interaction_fns: Sequence[Callable],  # TODO: name (everywhere)
) -> Callable[[ArrayLike, Sequence[ArrayLike]], Array]:
    # TODO: In fact it's questionable, whether a separate interaction_fn for
    #  each level is needed at all. `special_periodic_convolve` should work
    #  for all levels, shouldn't it?
    #  Do we still want to offer separate functions for increased flexibility?
    #  Even for restriction and prolongation a single function might be sufficient?
    if (
        not len(restriction_fns)
        == len(prolongation_fns)
        == len(interaction_fns)
    ):
        raise ValueError(
            "restriction_fns, prolongation_fns, interaction_fns "
            "must all have same length."
        )
    # TODO: Variable naming: Is `n_levels` appropriate here? We generally need
    #  to distinguish the highest splitting level and the highest level
    #  included in evaluation
    n_levels = len(restriction_fns) - 1

    # TODO: Name of the returned function?
    def grid_pass(
        gridcharge_lvl_one: ArrayLike, kernel_stencils: Sequence[ArrayLike]
    ) -> Array:
        if not len(kernel_stencils) == n_levels + 1:
            raise ValueError(
                "Wrong number of kernel stencils. "
                "Expected {} (including a placeholder at level zero), "
                "got {}.".format(n_levels + 1, len(kernel_stencils))
            )
        # TODO: list instead of dict?
        gridcharges_all_levels = {1: gridcharge_lvl_one}

        # TODO: More efficient to do the pass in the following order instead?
        #  1. Iteratively go up (compute grid charges at all levels)
        #  2. Apply interactions at all levels (possibly via `tree_map`)
        #  3. Iteratively go back down.

        # Go up ladder
        for lvl in range(2, n_levels + 1):
            restrict = restriction_fns[lvl]
            gridcharge_fine = gridcharges_all_levels[lvl - 1]
            gridcharge_coarse = restrict(gridcharge_fine)
            gridcharges_all_levels[lvl] = gridcharge_coarse

        # Apply top-level interaction
        gridcharge_toplevel = gridcharges_all_levels[n_levels]
        interact_toplevel = interaction_fns[n_levels]
        kernel_stencils_toplevel = kernel_stencils[n_levels]
        gridpotential = interact_toplevel(
            gridcharge_toplevel, kernel_stencils_toplevel
        )

        # Go down ladder
        for lvl in range(n_levels - 1, 0, -1):
            gridpotential = interaction_fns[lvl](
                gridcharges_all_levels[lvl], kernel_stencils[lvl]
            ) + prolongation_fns[lvl](gridpotential)

        return gridpotential

    return grid_pass


# TODO: Also add function for the calculation of energy by direct
#  contraction of grid charges with grid potential (without going the route
#  of reconstructing electrostatic potential by interpolation)

# TODO: Also add function for calculating gradient w.r.t. charge without
#  autodiffing the whole energy function?


def make_compute_longrange_energy(
    per_particle_basis_fn: BasisEvalFn,
    grid_pass_fn: Callable[[ArrayLike, ArrayLike], Array],
    grid_shape_lvl_one: tuple[int, ...],
    transform_mode: CellMode | None = None,
):
    def compute(positions, charges, kernel_stencils, cell=None):
        # TODO: Should cell really have a default?
        transform_pos, backtransform_grad = _make_unitcube_transform_fns(
            cell, transform_mode
        )
        per_particle_basis_vals, per_particle_inds = per_particle_basis_fn(
            transform_pos(positions)
        )
        gridcharge_lvl_one = _anterpolate(
            per_particle_basis_vals,
            per_particle_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        return _interpolate_energy(
            gridpotential_lvl_one,
            per_particle_basis_vals,
            per_particle_inds,
            charges,
        )

    return compute


def make_static_cell_longrange_fn(
    compute_longrange,  # TODO: argument name
    kernel_stencils,
    cell=None,
):
    # TODO: Should cell really default to None?
    def compute(positions, charges):
        return compute_longrange(
            positions, charges, cell=cell, kernel_stencils=kernel_stencils
        )

    return compute


def make_dyn_cell_longrange_fn(
    compute_longrange, kernel_stencil_construction_fn
):
    def compute(positions, charges, cell):
        return compute_longrange(
            positions,
            charges,
            cell=cell,
            kernel_stencils=kernel_stencil_construction_fn(cell),
        )

    return compute
