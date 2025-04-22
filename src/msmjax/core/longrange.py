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


def _anterpolate(
    basis_vals: ArrayLike,
    basis_inds: ArrayLike,
    charges: ArrayLike,
    grid_shape: tuple[int, ...],
):
    """Low-level function doing anterpolation (= calculating grid charge).

    Args:
        basis_vals: Array of shape `(n_particles, support_size)`, where
            `support_size` is the number of grid points around one particle
            with non-zero values of their basis functions. Contains the
            values of nearby non-zero basis functions for all particles. The
            first axis runs over particles, the second over grid points.
        basis_inds: Array of the same shape as ``basis_vals`` which, for all
            particles, contains the `flat` (!) indices of all nearby grid
            points with non-zero basis function values.
        charges: Array of charges, shape `(n_particles,)`.
        grid_shape: Tuple of integers representing shape of the target
            grid to which particle charges will be anterpolated.

    Returns:
        An array of the same shape as the ``grid_shape`` parameter that
        contains the value of the grid charge for each grid point.
    """
    grid_size = int(onp.prod(grid_shape))
    gridcharge_flat = jnp.zeros(grid_size)
    gridcharge_flat = gridcharge_flat.at[basis_inds].add(
        charges[:, jnp.newaxis] * basis_vals
    )
    return gridcharge_flat.reshape(grid_shape)


def _interpolate_energy(
    gridpotential: ArrayLike,
    basis_vals: ArrayLike,
    basis_inds: ArrayLike,
    charges: ArrayLike,
) -> Array:
    """Low-level function calculating long-range energy from grid potential.

    Args:
        gridpotential: Array of grid potential (:math:`e^{l+}` in the language
            of the reference).
        basis_vals: See :func:`_anterpolate`.
        basis_inds: See :func:`_anterpolate`.
        charges: Array of particle charges, shape `(n_particles,)`.

    Returns:
        The scalar electrostatic energy.
    """
    energy = 0.5 * jnp.sum(
        charges * (gridpotential.take(basis_inds) * basis_vals).sum(axis=1)
    )
    return energy


def _interpolate_energy_positions_gradient(
    gridpotential: ArrayLike,
    basis_grads: ArrayLike,
    basis_inds: ArrayLike,
    charges: ArrayLike,
) -> Array:
    """Low-level function calculating positions gradient of long-range energy.

    Implements an analytic expression for the derivative that calculates it
    by explicitly interpolating it from the grid potential. This allows a
    more efficient computation than default automatic differentiation of
    the energy.

    Args:
        gridpotential: Array of grid potential (:math:`e^{l+}` in the language
            of the reference).
        basis_grads: Similar to ``basis_vals`` (see :func:`_anterpolate`),
            but containing the gradients of the basis functions w.r.t.
            particle positions instead of their values. Shape `(n_particles,
            support_size, n_dim)`, where `n_dim` is the spatial dimension of
            the system.
        basis_inds: See :func:`_anterpolate`.
        charges: Array of particle charges, shape `(n_particles,)`.

    Returns:
        The gradient of the long-range energy w.r.t. particle positions,
        which is an array of shape `(n_particles, n_dim)`.
    """
    result = charges[:, jnp.newaxis] * jnp.sum(
        gridpotential.take(basis_inds)[..., jnp.newaxis] * basis_grads, axis=1
    )
    return result


def _interpolate_energy_charge_gradient(
    gridpotential: ArrayLike, basis_vals: ArrayLike, basis_inds: ArrayLike
) -> Array:
    """Low-level function calculating charge gradient of long-range energy.

    Implements an analytic expression for the derivative that calculates it
    by explicitly interpolating it from the grid potential. This allows a
    more efficient computation than default automatic differentiation of
    the energy.

    # TODO: mention the relation to the electrostatic potential at particle positions?

    Args:
        gridpotential: Array of grid potential (:math:`e^{l+}` in the language
            of the reference).
        basis_vals: See :func:`_anterpolate`.
        basis_inds: See :func:`_anterpolate`.

    Returns:
        The gradient of the long-range energy w.r.t. particle charges,
        which is an array of shape `(n_particles,)`.
    """
    return (gridpotential.take(basis_inds) * basis_vals).sum(axis=1)


@partial(jax.jit, static_argnames=["pbc", "method"])
def special_periodic_convolve(
    data: ArrayLike,
    kernel: ArrayLike,
    pbc: Sequence[bool],
    method: Literal["direct", "fft"],  # TODO: centralize definition? default?
) -> Array:
    """Perform a specialized case of convolution with optional wrapping.

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
            :func:`jax.scipy.signal.convolve`. `fft` is usually much faster.

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


def make_grid_pass_fn(
    restriction_fns: Sequence[Callable[[ArrayLike], Array]],
    prolongation_fns: Sequence[Callable[[ArrayLike], Array]],
    interaction_fns: Sequence[
        Callable[[ArrayLike, ArrayLike], Array]
    ],  # TODO: name (everywhere)
) -> Callable[[ArrayLike, Sequence[ArrayLike | None]], Array]:
    """Create a function that makes a pass through all grid levels.

    In other words, create the linear operator (consisting of restriction of
    the grid charge to higher grid levels, calculation of potentials at all
    levels, and prolongation of the higher-level potentials down to lower
    levels) that connects the grid charge at level one to the accumulated
    grid potential at level one. This corresponds to the upper part of the
    V-cycle diagram as which the MSM is commonly visualized.

    .. note::

       The operator sequences passed as parameters to this function need to
       respect the grid-level indexing convention. As per convention,
       the index 0 refers to the particle level (where interactions are
       computed directly without grids), whereas the lowest actual grid
       level is at index 1. Further, for the restriction and prolongation
       functions, which connect two different grid levels, the convention is
       that the function whose `output` lives on grid level :math:`l` is
       located at index :math:`l` of the sequence. For example, the operator
       :math:`\mathcal{ I}^2_1` that restricts the grid charge from level 1
       to 2 would be addressed as ``restriction_fns[2]``.

       Thus, the input for, e.g., four grid levels should look like this,
       employing placeholders where needed:

       .. code-block:: python

          restriction_fns = [None, None, I_21, I_32, I_43]
          prolongation_fns = [None, I_12, I_23, I_34, None]
          interaction_fns = [None, K_1, K_2, K_3, K_4]

    Args:
        restriction_fns: Sequence of restriction functions,
            one for each level, including placeholders (see above note).
            Corresponding to the upward arrows on the left side of the
            V-cycle diagram. Input is array of grid charge, output is array
            of grid charge one level higher.
        prolongation_fns: Sequence of prolongation functions,
            one for each level, including placeholders (see above note).
            Corresponding to the downward arrows on the right side of the
            V-cycle diagram. Input is array of grid potential,
            output is array of potential prolongated to the next lower level.
        interaction_fns: Sequence of interaction functions (TODO: name),
            one for each level, including placeholders (see above note).
            Corresponding to the horizontal arrows in the V-cycle diagram.
            Inputs are two arrays, grid charge and a kernel coefficient
            stencil, output is the grid potential on the same level.
            Mathematically:
            :math:`e^{l}_{\\mathbf{m}} = \sum_{\mathbf{n}} K^l_{\\mathbf{m}
            - \\mathbf{n}} \\tilde{q}^l_{\\mathbf{n}}`.

    Returns:
        A function for performing the pass through all grid levels.
        It calculates the accumulated grid potential :math:`e^{1+}` at level
        one and takes two arguments:

            - The level-one grid charge :math:`\\tilde{q}^1`.
            - A sequence of coefficient stencil arrays for the interaction
              kernels, one per grid level (including a placeholder at level
              zero, see note above on grid-level indexing convention). These
              are passed to the ``interaction_fns`` that were supplied to
              construct the grid pass function. The :math:`l`-th stencil is
              consumed by the :math:`l`-th element of ``ìnteraction_fns``
    """
    # TODO: Don't take interaction_fns as a parameter at all and default to
    #  always using special_periodic_convolve? In this case, pbc and conv_meth
    #  would need to be added as parameters.
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


def make_compute_u_oneplus(
    singleparticle_basis_fn_lvl_one: Callable[
        [ArrayLike], tuple[Array, Array]
    ],
    grid_pass_fn: Callable[[ArrayLike, Sequence[ArrayLike | None]], Array],
    grid_shape_lvl_one: tuple[int, ...],
    transform_mode: CellMode | None = None,  # TODO: name
    kernel_stencils: Sequence[None | ArrayLike] = None,
    kernel_stencil_construction_fn: Callable[
        [ArrayLike], Sequence[ArrayLike | None]
    ] = None,
    use_custom_derivatives: bool = True,
) -> Callable[[ArrayLike, ArrayLike, Sequence[ArrayLike | None]], Array]:
    """Create a function that computes the MSM long-range energy contribution.

    The quantity being (approximately) calculated is called :math:`U^{1+}`
    in the reference article.

    Args:
        singleparticle_basis_fn_lvl_one:
            A function that, for one particle,

                1) identifies all points on the level-one grid that are
                   sufficiently close for the particle's position to be
                   contained within the support of the associated basis
                   functions, i.e., finds the set of grid points
                   :math:`M = \\{
                   \\mathbf{m} : \\varphi^{1}_{\\mathbf{m}}(\\mathbf{r}_i) \\neq 0
                   \\} \\,`
                   (where :math:`\\mathbf{r}_i` denotes the position of
                   particle :math:`i` and
                   :math:`\\mathbf{\\varphi^{1}_{\\mathbf{m}}}` is the
                   basis function centered on point :math:`\\mathbf{m}` of the
                   level-one grid),

                2) evaluates the corresponding basis functions, i.e.
                   computes :math:`\\varphi^{1}_{\\mathbf{m}}(\\mathbf{r}_i)`
                   for all grid points :math:`\mathbf{m} \in M \\,`.

            Inputs and outputs:

                - Input to ``singleparticle_basis_fn_lvl_one`` should be a
                  1-d array, shape `(n_dim,)`, representing the coordinates
                  of a single particle.

                - Output of ``singleparticle_basis_fn_lvl_one`` should be a
                  tuple of two 1-d arrays, each of shape `(support_size,)`,
                  where `support_size` designates the number of non-zero
                  basis functions around one particle (= the cardinality of
                  :math:`M` from above). The first of the two arrays
                  contains the values of the basis functions at all grid
                  points :math:`\mathbf{m} \in M`. The second array contains
                  the corresponding set of grid point indices :math:`M` as
                  `flat` (!) indices into the grid.

            .. note::
               Regardless of the spatial dimension of the system,
               ``singleparticle_basis_fn_lvl_one`` should always return flat
               arrays.

            .. warning::
               If ``singleparticle_basis_fn_lvl_one`` returs indices that
               are out of bounds w.r.t. to the grid size defined by
               ``grid_shape_lvl_one``, this will result in NaNs. This is
               intentional because errors like particles moving outside
               the grid boundaries might otherwise go unnoticed.

        grid_pass_fn: A function that performs the entire process of moving up,
            across, and back down the grid hierarchy. For more details on
            the expected signature, see :func:`make_grid_pass_fn`, which can
            be used conveniently to create such a function.

            Inputs and outputs:

                - Input to ``grid_pass_fn`` should be the level-one grid
                  charge :math:`\\tilde{q}^1` (an array whose shape matches
                  the ``grid_shape_lvl_one`` parameter), and a sequence of
                  coefficient stencils for the interaction kernels (one per
                  grid level, including a placeholder at level zero).
                - Output of ``grid_pass_fn`` should be the level-one grid
                  potential, also of shape ``grid_shape_lvl_one``.

        grid_shape_lvl_one: Tuple of integers representing shape of target
            grid at level one, to which particle charges will be anterpolated.
        use_custom_derivatives: Whether the returned energy function should
            use custom (more efficient) differentiation rules for its
            derivatives w.r.t. positions and charges.

    Returns:
        A function that computes the scalar energy :math:`U^{1+}` and takes
        three arguments:

        - Array of positions, shape `(n_particles, n_dim)`.
        - Array of charges, shape `(n_particles,)`.
        - A sequence of coefficient stencils for the interaction kernels (one
          per grid level, including a placeholder at level zero). Passed to
          ``grid_pass_fn``. See :func:`make_grid_pass_fn` for more details.
    """

    if kernel_stencils is None and kernel_stencil_construction_fn is None:
        raise ValueError(
            "One of kernel_stencils or kernel_stencil_construction_fn "
            "is required."
        )
    if (
        kernel_stencils is not None
        and kernel_stencil_construction_fn is not None
    ):
        raise ValueError(
            "kernel_stencils and kernel_stencil_construction_fn "
            "are mutually exclusive."
        )

    def _calc_energy(
        positions: ArrayLike,
        charges: ArrayLike,
        kernel_stencils: Sequence[ArrayLike | None],
    ) -> Array:
        """Compute the MSM's long-range energy contribution :math:`U^{1+}`.

        Args:
            positions: Array of positions, shape `(n_particles, n_dim)`.
            charges: Array of charges, shape `(n_particles,)`.
            kernel_stencils: A sequence of coefficient stencils for the
                interaction kernels (one per grid level, including a
                placeholder at level zero). Passed to ``grid_pass_fn``. See
                :func:`make_grid_pass_fn` for more details.

        Returns:
            The long-range energy contribution :math:`U^{1+}`.
        """
        basis_vals, basis_inds = jax.vmap(singleparticle_basis_fn_lvl_one)(
            positions
        )
        gridcharge_lvl_one = _anterpolate(
            basis_vals, basis_inds, charges, grid_shape_lvl_one
        )
        gridpotential_lvl_oneplus = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        return _interpolate_energy(
            gridpotential_lvl_oneplus, basis_vals, basis_inds, charges
        )

    @jax.custom_jvp
    def _calc_energy_custom(
        positions: ArrayLike,
        charges: ArrayLike,
        kernel_stencils: Sequence[ArrayLike],
    ) -> Array:
        """Compute the long-range energy contribution :math:`U^0` using custom
        derivative rules.

        See ``_calc_energy`` for parameter details.
        """
        return _calc_energy(positions, charges, kernel_stencils)

    @_calc_energy_custom.defjvp
    def _calc_energy_custom_jvp(primals, tangents):
        """Defines custom derivative rules for _calc_energy_custom"""
        (positions, charges, kernel_stencils) = primals
        (positions_dot, charges_dot, kernel_stencils_dot) = tangents

        # Energy
        basis_vals, basis_inds = jax.vmap(singleparticle_basis_fn_lvl_one)(
            positions
        )
        gridcharge_lvl_one = _anterpolate(
            basis_vals, basis_inds, charges, grid_shape_lvl_one
        )
        gridpotential_lvl_oneplus = grid_pass_fn(
            gridcharge_lvl_one, kernel_stencils
        )
        energy = _interpolate_energy(
            gridpotential_lvl_oneplus, basis_vals, basis_inds, charges
        )

        # Derivative w.r.t. positions:
        basis_grads, basis_inds = jax.vmap(
            jax.jacfwd(singleparticle_basis_fn_lvl_one, has_aux=True)
        )(positions)
        positions_jac = _interpolate_energy_positions_gradient(
            gridpotential_lvl_oneplus, basis_grads, basis_inds, charges
        )
        positions_tangent_out = (positions_jac * positions_dot).sum()

        # Derivative w.r.t. charges:
        charges_jac = _interpolate_energy_charge_gradient(
            gridpotential_lvl_oneplus, basis_vals, basis_inds
        )
        charges_tangent_out = (charges_jac * charges_dot).sum()

        # Derivative w.r.t. kernel stencils:
        # In contrast to the positions and charges, we cannot supply a
        # custom derivative rule for this parameter, as the functional form
        # of `grid_pass_fn` is unspecified. Therefore, we fall back to
        # default automatic differentiation. This is done in a slightly
        # hacky way, by calling the regular jvp, but with the input tangents
        # corresponding to all parameters except ``kernel_stencils`` set to
        # zero.
        tangents_zeroed = (
            onp.zeros(positions.shape, dtype=float),  # TODO: jnp?
            onp.zeros(charges.shape, dtype=float),  # TODO: jnp?
            kernel_stencils_dot,
        )
        _, kernel_stencils_tangent_out = jax.jvp(
            _calc_energy, primals, tangents_zeroed
        )

        primal_out = energy
        tangent_out = (
            positions_tangent_out
            + charges_tangent_out
            + kernel_stencils_tangent_out
        )

        return primal_out, tangent_out

    def _make_unitcube_transform_fn(
        cell: ArrayLike,
    ) -> Callable[[ArrayLike], Array]:
        if transform_mode == "ortho":
            inverse = 1.0 / jnp.diag(cell)
            return lambda x: x * inverse
        elif transform_mode == "general":
            inverse = jnp.linalg.pinv(cell)
            return lambda x: x @ inverse
        else:
            raise ValueError(f"Invalid 'transform_mode': {transform_mode}")

    def compute_u_oneplus(
        positions: ArrayLike,
        charges: ArrayLike,
        cell: ArrayLike,
    ) -> Array:
        # TODO: custom derivatives yes or no
        # TODO: transform positions yes or no
        # TODO: kernel_stencils or kernel_stencil_construction_fn
        # TODO: make cell optional?
        if transform_mode is not None:
            positions_to_unitcube = _make_unitcube_transform_fn(cell)
            positions = positions_to_unitcube(positions)

        if kernel_stencils is not None:
            stencils = kernel_stencils
        else:
            stencils = kernel_stencil_construction_fn(cell)

        if not use_custom_derivatives:
            return _calc_energy(positions, charges, stencils)

        return _calc_energy_custom(positions, charges, stencils)

    return compute_u_oneplus
