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


def _interpolate_potential():
    # TODO: Do we want to provide this?
    pass


def _interpolate_energy(
    gridpotential: ArrayLike,
    basis_vals: ArrayLike,
    indices: ArrayLike,
    charges: ArrayLike,
) -> Array:
    """Low-level function for calculating the energy from the grid potential.

    Args:
        gridpotential: Array of grid potential (:math:`e^{l+}` in the language
            of the reference).
        basis_vals: TODO: How best to document (appears in several places)? More informative variable name? (`per_particle_basis_vals`?)
        indices: TODO: How best to document (appears in several places)? More informative variable name? (`per_particle_inds`?)
        charges: Array of particle charges, shape `(n_particles,)`.

    Returns:
        The scalar electrostatic energy.
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


def _make_unitcube_transform_fns(
    cell: ArrayLike | None, transform_mode: CellMode | None
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array]]:
    # TODO: Is this the right module for this function?
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
    # TODO: Should this function be a protected member?
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


# TODO: This is not needed anymore, is it?
def make_anterpolation_fn(
    per_particle_basis_fn: BasisEvalFn, grid_shape: tuple[int, ...]
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """

    Args:
        per_particle_basis_fn: A function that, for all particles, identifies
            those grid points with non-zero basis function values,
            and evaluates them. See :func:`make_compute_longrange_energy`
            for details of signature and meaning of return values.
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
        basis_vals, indices = per_particle_basis_fn(positions)
        gridcharge_flat = jnp.zeros(grid_size)
        gridcharge_flat = gridcharge_flat.at[indices].add(
            charges[:, jnp.newaxis] * basis_vals
        )
        return gridcharge_flat.reshape(grid_shape)

    return anterpolate


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


# TODO: Function name? Should it include `oneplus` somehow?
def make_compute_longrange_energy(
    per_particle_basis_fn: BasisEvalFn,
    grid_pass_fn: Callable[[ArrayLike, Sequence[ArrayLike]], Array],
    grid_shape_lvl_one: tuple[int, ...],
    transform_mode: CellMode | None = None,
):
    """Create a function that computes the energy by interpolating potential.

    The quantity being (approximately) calculated is called :math:`U^{1+}`
    in the reference article.

    Args:
        per_particle_basis_fn: A function that, for each particle,

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

                - Input to ``per_particle_basis_fn`` should be a 2-d array of
                  particle positions, shape `(n_particles, n_dim)`.

                - Output of ``per_particle_basis_fn`` should be a tuple of two
                  2-d arrays, each of shape `(n_particles, support_size)`,
                  where `support_size` designates the fixed number of
                  non-zero basis functions around each particle (= the
                  cardinality of :math:`M` from above).
                  Their first axes run over particles, and the second over
                  grid points.
                  The first of the two arrays contains the values of the basis
                  functions for each particle, and the second array contains
                  the `flat` (!) indices of the corresponding grid points.

        grid_pass_fn: A function that performs the entire moving up,
            across, and back down the grid hierarchy.

            Inputs and outputs:

                - Input to ``grid_pass_fn`` should be the level-one grid
                  charge :math:`\\tilde{q}^1` (an array of shape equal to
                  the ``grid_shape_lvl_one`` parameter), and a sequence of
                  coefficient stencils :math:`\\mathcal{K}^l` for the
                  interaction kernels (one per grid level, including a
                  placeholder at level zero).
                - Output of ``grid_pass_fn`` should be the level-one grid
                  potential, of shape ``grid_shape_lvl_one``.

        grid_shape_lvl_one: Tuple of integers indicating the shape of the
            target grid to which to anterpolate the particle charges.
        transform_mode: TODO: A string specifying assumptions on the shape of the
            unit cell. Either the cell is assumed orthorhombic and
            axis-aligned, in which case only its diagonal is considered,
            reducing computational cost, or a general triclinic one.
            May be omitted if the cell is both orthorhombic and static.

    Returns:
        TODO
    """

    def compute(positions, charges, kernel_stencils, cell=None):
        # TODO: Should cell really have a default? Watch out for interaction
        #  between defaults of transform_mode and cell.
        transform_pos, backtransform_grad = _make_unitcube_transform_fns(
            cell, transform_mode
        )
        basis_vals, basis_inds = per_particle_basis_fn(
            transform_pos(positions)
        )
        # TODO: The next two statements are repeated in every,
        #  make_compute_longrange_something function. Should they be wrapped
        #  in a single function?
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        return _interpolate_energy(
            gridpotential_lvl_one,
            basis_vals,
            basis_inds,
            charges,
        )

    return compute


# TODO: Function name? Should it include `oneplus` somehow?
def make_compute_longrange_forces(
    per_particle_basis_and_grad_fn: BasisEvalFn,
    grid_pass_fn: Callable[[ArrayLike, ArrayLike], Array],
    grid_shape_lvl_one: tuple[int, ...],
    transform_mode: CellMode | None = None,
):
    def compute(positions, charges, kernel_stencils, cell=None):
        # TODO: Should cell really have a default?
        transform_pos, backtransform_grad = _make_unitcube_transform_fns(
            cell, transform_mode
        )
        (basis_vals, basis_inds), basis_grads = per_particle_basis_and_grad_fn(
            transform_pos(positions)
        )
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        forces = _interpolate_forces(
            gridpotential_lvl_one,
            basis_grads,
            basis_inds,
            charges,
        )
        return backtransform_grad(forces)

    return compute


# TODO: Function name? Should it include `oneplus` somehow?
def make_compute_longrange_energy_and_forces(
    per_particle_basis_and_grad_fn: BasisEvalFn,
    grid_pass_fn: Callable[[ArrayLike, ArrayLike], Array],
    grid_shape_lvl_one: tuple[int, ...],
    transform_mode: CellMode | None = None,
):
    def compute(positions, charges, kernel_stencils, cell=None):
        # TODO: Should cell really have a default?
        transform_pos, backtransform_grad = _make_unitcube_transform_fns(
            cell, transform_mode
        )
        (basis_vals, basis_inds), basis_grads = per_particle_basis_and_grad_fn(
            transform_pos(positions)
        )
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        energy = _interpolate_energy(
            gridpotential_lvl_one,
            basis_vals,
            basis_inds,
            charges,
        )
        forces = _interpolate_forces(
            gridpotential_lvl_one,
            basis_grads,
            basis_inds,
            charges,
        )
        return energy, backtransform_grad(forces)

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


# TODO: If all is reduced to this custom-derivative energy function and there
#  are no more separate force functions, there is really no reason not to call
#  it compute_u_oneplus or similar
def make_compute_energy_customjvp(
    charges, kernel_stencils, grid_shape_lvl_one, single_particle_basis_fn
):
    # - Hardcoding charges, kernel stencils for now to not have to deal with
    #   multiple parameters in JVP definition.
    # - Hardcoding the grid shape for now.
    # - Hardcoding the one-particle basis eval function for now.

    @jax.custom_jvp
    def compute_energy(positions):
        basis_vals, basis_inds = jax.vmap(single_particle_basis_fn)(positions)
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        return _interpolate_energy(
            gridpotential_lvl_one,
            basis_vals,
            basis_inds,
            charges,
        )

    @compute_energy.defjvp
    def compute_energy_jvp(primals, tangents):
        (positions,) = primals
        (positions_t,) = tangents
        basis_vals, basis_inds = jax.vmap(single_particle_basis_fn)(positions)
        # TODO: don't repeat basis_inds assignment
        # TODO: Order of jac and vmap? jacfwd/jacrev? (jacrev appears much slower in a quick test)
        basis_grads, basis_inds = jax.vmap(
            jax.jacfwd(single_particle_basis_fn, has_aux=True)
        )(positions)
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        # TODO: can we compute the energy by calling compute_e_jvp, or does that ruin the efficiency?
        energy = _interpolate_energy(
            gridpotential_lvl_one,
            basis_vals,
            basis_inds,
            charges,
        )
        # TODO: don't go via forces, which requires double minus
        forces = _interpolate_forces(
            gridpotential_lvl_one,
            basis_grads,
            basis_inds,
            charges,
        )
        primals_out = energy
        tangents_out = -(forces * positions_t).sum()  # TODO: minus (see above)

        return primals_out, tangents_out

    return compute_energy


def make_compute_u_oneplus(
    single_particle_basis_fn: BasisEvalFn,
    grid_pass_fn: Callable[[ArrayLike, Sequence[ArrayLike]], Array],
    grid_shape_lvl_one: tuple[int, ...],
):
    def _compute_u_oneplus(
        positions: ArrayLike,
        charges: ArrayLike,
        kernel_stencils: Sequence[ArrayLike],
    ):
        basis_vals, basis_inds = jax.vmap(single_particle_basis_fn)(positions)
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        # TODO: Do direct contraction of gridcharge and gridpotential instead
        #  of interpolating back?
        return _interpolate_energy(
            gridpotential_lvl_one,
            basis_vals,
            basis_inds,
            charges,
        )

    @jax.custom_jvp
    def compute_u_oneplus(
        positions: ArrayLike,
        charges: ArrayLike,
        kernel_stencils: Sequence[ArrayLike],
    ):
        return _compute_u_oneplus(positions, charges, kernel_stencils)

    def compute_u_oneplus_dx(
        positions_dot, primal_out, positions, charges, kernel_stencils
    ):
        basis_vals, basis_inds = jax.vmap(single_particle_basis_fn)(positions)
        # TODO: Order of jac and vmap? jacfwd/jacrev?
        #  (jacrev was much slower in a quick test, which also makes sense
        #  because single_particle_fn has (in 3 dimensions) 3 inputs vs.
        #  something on the order of 64 or 216 outputs)
        basis_grads, _ = jax.vmap(
            jax.jacfwd(single_particle_basis_fn, has_aux=True)
        )(positions)
        gridcharge_lvl_one = _anterpolate(
            basis_vals,
            basis_inds,
            charges,
            grid_shape_lvl_one,
        )
        gridpotential_lvl_one = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        # TODO: don't go via forces, which requires double minus
        forces = _interpolate_forces(
            gridpotential_lvl_one,
            basis_grads,
            basis_inds,
            charges,
        )
        # TODO: minus (see above)
        return -(forces * positions_dot).sum()

    def compute_u_oneplus_dq(
        charges_dot, primal_out, positions, charges, kernel_stencils
    ):
        primals_in = (positions, charges, kernel_stencils)
        # TODO: kernel_stencils is not an array but a sequence of arrays with
        #  None as the first element?
        tangents_in = (
            onp.zeros_like(positions),
            charges_dot,
            onp.zeros_like(kernel_stencils),
        )
        _, tangents_out = jax.jvp(_compute_u_oneplus, primals_in, tangents_in)
        return tangents_out

    def compute_u_oneplus_dk(
        kernel_stencils_dot, primal_out, positions, charges, kernel_stencils
    ):
        pass

    compute_u_oneplus.defjvps(
        compute_u_oneplus_dx, compute_u_oneplus_dq, compute_u_oneplus_dk
    )

    return compute_u_oneplus
