"""Code for short-range part (that is directly evaluated, without grids).

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
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from msmjax.jax_md import space
from msmjax.utils import _divide_zero_safe

# TODO: This should be defined elsewhere, since the longrange part will likely
#  also use it
CellMode = Literal["ortho", "general"]
KernelFn = Callable[[ArrayLike], Array]


def _gen_supercell(
    positions: ArrayLike,
    charges: ArrayLike,
    cell: ArrayLike,
    supercell_diag: Sequence[int],
) -> tuple[Array, Array, Array]:
    """Replicate unit cell and contained particles along its axes.

    Args:
        positions: Array of positions, shape `(n_particles, n_dim)`.
        charges: Array of charges, shape `(n_particles,)`.
        cell: Array representing unit cell, shape `(n_dim, n_dim)`.
        supercell_diag: Sequence of positive integers, one for each direction,
            indicating the number of times to replicate the system.

    Returns:
        Tuple containing
            - array of positions after replication,
            - array of charges after replication,
            - unit cell after replication.
    """
    n_particles, n_dim = positions.shape
    M = onp.prod(supercell_diag)
    tile_positions = jnp.tile(positions, (M, 1))
    super_charges = jnp.tile(charges, M)
    grid = jnp.indices(supercell_diag).reshape(n_dim, -1).T
    translations = jnp.dot(grid, cell)
    tile_translations = jnp.repeat(translations, n_particles, axis=0)
    super_positions = tile_positions + tile_translations
    super_cell = cell * jnp.array(supercell_diag)[:, jnp.newaxis]
    return super_positions, super_charges, super_cell


def _generalized_diagonal_mask(a: ArrayLike) -> Array:
    """Set the diagonal of a, possibly wider than tall, matrix to zero.

    Adapted from JAX-MD. # TODO: JAX-MD attribution

    .. warning::
       Any NaN or infinite entries (including ones off the diagonal!) will
       be silently replaced by this function. For the diagonal, this is
       usually reasonable and desired. That is because in the case for which
       this function is designed, diagonal elements of the input correspond
       to interactions of particles with themselves, which are usually
       considered artifactual and which may be undefined. When off-diagonal
       elements are replaced this way, however, this may obscure the origin
       of bugs that caused them to be invalid.

    Args:
        a: Original matrix.

    Returns:
        The matrix with diagonal set to zero.
    """
    if len(a.shape) != 2:
        raise ValueError("Only two-dimensional arrays are supported.")
    M, N = a.shape
    if M > N:
        raise ValueError(
            "Input array must be either square, or wider than tall."
        )
    a = jnp.nan_to_num(a)
    mask = 1.0 - jnp.eye(M, dtype=a.dtype)
    mask = jnp.pad(
        mask,
        pad_width=((0, 0), (0, N - M)),
        mode="constant",
        constant_values=1,
    )
    return mask * a


def _displacement_free(r_1: ArrayLike, r_2: ArrayLike) -> Array:
    """Compute distance vector between two points in free space

    Args:
        r_1: First point
        r_2: Second point

    Returns:
        Distance vector.
    """
    # TODO: unit test this on its own?
    return r_1 - r_2


def _displacement_ortho(
    r_1: ArrayLike, r_2: ArrayLike, side_lengths: ArrayLike
) -> Array:
    """Compute distance vector between two points in orthorhombic cell.

    Handles mixed periodicity: To indicate that specific directions lack
    periodicity, set the corresponding elements of ``side_lengths`` to zero.
    Periodicity is accounted for by means of the minimum image convention,
    with all the known limitations entailed by this.

    Args:
        r_1: First point
        r_2: Second point
        side_lengths: 1-d array of side lengths (one per direction)

    Returns:
        Distance vector.
    """
    # TODO: unit test this on its own?
    delta = r_1 - r_2
    return (
        delta
        - jnp.round(_divide_zero_safe(delta, side_lengths)) * side_lengths
    )


def _displacement_general(
    r_1: ArrayLike, r_2: ArrayLike, cell: ArrayLike
) -> Array:
    """Compute distance vector between two points in general triclinic cell.

    Handles mixed periodicity: To indicate that specific directions lack
    periodicity, set the corresponding rows of ``cell`` to zero.
    Periodicity is accounted for by means of the minimum image convention,
    with all the known limitations entailed by this.

    Args:
        r_1: First point
        r_2: Second point
        cell: Array representing unit cell, shape `(n_dim, n_dim)`.

    Returns:
        Distance vector.
    """
    # TODO: unit test this on its own?
    dr = r_1 - r_2
    inv_cell = jnp.linalg.pinv(cell)
    r_1_transf = r_1 @ inv_cell
    r_2_transf = r_2 @ inv_cell
    dr_transformed = r_1_transf - r_2_transf
    return dr - jnp.round(dr_transformed) @ cell


def _concretize_displacement_fn(
    pbc: Sequence[bool],
    cell_mode: Optional[CellMode] = None,
) -> Callable[[ArrayLike, ArrayLike, Optional[ArrayLike]], Array]:
    """Select/construct displacement fn based on PBCs, unit cell constraints.

    Wraps lower-level displacement functions and transforms them into ones
    with a choice of periodic boundary conditions built-in already, and with
    a consistent signature.

    Args:
        pbc: One boolean per direction signaling periodicity.
        cell_mode: A string specifying assumptions on the shape of the
            unit cell. Either the cell is assumed orthorhombic and
            axis-aligned, in which case only its diagonal is considered,
            reducing computational cost, or a general triclinic one. May be
            omitted (and is ignored) if no direction is periodic.

    Returns:
        A function of two position vector arguments and (optionally, depending
        on PBCs) a unit cell, that computes the distance vector between them.
    """
    if onp.any(pbc) and cell_mode is None:
        # TODO: write test for this check
        raise ValueError(
            "If at least one direction is periodic, "
            "you must specify cell_mode."
        )

    if not onp.any(pbc):

        def displacement_fn(r_1, r_2, cell=None):
            return _displacement_free(r_1, r_2)

        return displacement_fn

    if cell_mode == "ortho":

        def displacement_fn(r_1, r_2, cell):
            side_lengths_processed_for_pbc = jnp.diag(cell) * pbc
            return _displacement_ortho(
                r_1, r_2, side_lengths_processed_for_pbc
            )

        return displacement_fn

    elif cell_mode == "general":

        def displacement_fn(r_1, r_2, cell):
            cell_processed_for_pbc = cell * pbc[:, jnp.newaxis]
            return _displacement_general(r_1, r_2, cell_processed_for_pbc)

        return displacement_fn

    else:
        raise ValueError("Invalid cell_mode.")


def make_eval_pair_pot(
    kernel_fn: KernelFn,
    pbc: Sequence[bool],
    cell_mode: Optional[CellMode] = None,
    supercell_diag: Optional[Sequence[int]] = None,
) -> Callable[[ArrayLike, ArrayLike, Optional[ArrayLike]], Array]:
    """Transform interaction kernel into function acting on a particle system.

    In other words, given a distance-dependent interaction kernel :math:`k(r)`,
    construct another function that computes the total system energy,
    :math:`\\frac{1}{2} \\sum_i \\sum_{j \\neq i} q_i q_j k(r_{ij})`,
    by mapping :math:`k(r)` over all particle pairs.

    .. warning::
       Distance computations under periodic boundary conditions are handled
       by means of the minimum image convention, with the known limitations
       this entails. If the cutoff radius of ``kernel_fn`` is too large for
       the unit cell, or the unit cell is too deformed, results will be
       incorrect. If you know beforehand that this is an issue, you can
       remedy it by using the ``supercell_diag`` parameter (see below).

    Args:
        kernel_fn: A function of a single scalar distance argument,
            corresponding to :math:`k(r)` in the above formula.
        pbc: One boolean per direction signaling periodicity.
        cell_mode: A string specifying assumptions on the shape of the
            unit cell. Either the cell is assumed orthorhombic and
            axis-aligned, in which case only its diagonal is considered,
            reducing computational cost, or a general triclinic one. May be
            omitted (and is ignored) if no direction is periodic.
        supercell_diag: An optional sequence of positive integers, one per
            direction. If supplied, pairwise interactions are computed
            between the particles in the original cell and all particles in
            a supercell created by repeating the cell the given number of
            times along each direction. This can be used to ensure that all
            interactions with neighbors are taken into account in cases
            where the cutoff of ``kernel_fn`` is too large for the
            original, non-replicated, cell.

    Returns:
        A function that takes arrays of particle positions and charges,
        and the unit cell, as arguments and computes the energy for the
        whole system of particles.
    """
    pbc = onp.asarray(pbc)
    if supercell_diag is None:
        supercell_diag = onp.ones_like(pbc, dtype=int)
    if onp.logical_and(~pbc, onp.asarray(supercell_diag) != 1).any():
        raise ValueError(
            "`supercell_diag` must be equal to one along non-periodic axes"
        )
    # TODO: check supercell_diag >= 1?

    displacement_fn = _concretize_displacement_fn(pbc, cell_mode)

    def compute_energy(
        positions: ArrayLike, charges: ArrayLike, cell: ArrayLike = None
    ) -> Array:
        """Evaluate pair potential for entire system of charged particles.

        Args:
            positions: Array of positions, shape `(n_particles, n_dim)`.
            charges: Array of charges, shape `(n_particles,)`.
            cell: Array representing unit cell, shape `(n_dim, n_dim)`.

        Returns:
            Total system energy.
        """
        # TODO: JAX-MD attribution
        # TODO: test calling without `cell` argument in non-periodic case
        if pbc.any():
            if cell is None:
                raise ValueError(
                    "If at least one direction is periodic, "
                    "the cell argument is required."
                )
            super_positions, super_charges, super_cell = _gen_supercell(
                positions=positions,
                charges=charges,
                cell=cell,
                supercell_diag=supercell_diag,
            )
            metric_fn = partial(space.metric(displacement_fn), cell=super_cell)
        else:
            super_positions, super_charges = positions, charges
            metric_fn = partial(space.metric(displacement_fn))
        mapped_metric_fn = space.map_product(metric_fn)
        dr_ij = mapped_metric_fn(super_positions, positions)
        qi_qj = charges[:, jnp.newaxis] * super_charges
        return (
            0.5 * (qi_qj * _generalized_diagonal_mask(kernel_fn(dr_ij))).sum()
        )

    return compute_energy


def make_eval_pair_pot_neighborlist(
    kernel_fn: KernelFn,
    pbc: Sequence[bool],
    cell_mode: Optional[CellMode] = None,
    safe_eval_distance: float = 1.0,  # TODO: test? (how?)
) -> Callable[
    [
        ArrayLike,
        ArrayLike,
        tuple[ArrayLike, ArrayLike],
        ArrayLike,
        Optional[ArrayLike],
    ],
    Array,
]:
    """Transform interaction kernel into function acting on a particle system.

    Like :func:`make_eval_pair_pot`, but with a neighbor list.

    .. warning::
       Distance computations under periodic boundary conditions are handled
       by means of the minimum image convention, with the known limitations
       this entails. If the cutoff radius of ``kernel_fn`` is too large for
       the unit cell, or the unit cell is too deformed, results will be
       incorrect. Unlike :func:`make_eval_pair_pot`, this neighbor-list
       version does not have a built-in supercell generation feature.

    Args:
        kernel_fn: A function of a single scalar distance argument,
            see documentation of :func:`make_eval_pair_pot`.
        pbc: One boolean per direction signaling periodicity.
        cell_mode: A string specifying assumptions on the shape of the
            unit cell. Either the cell is assumed orthorhombic and
            axis-aligned, in which case only its diagonal is considered,
            reducing computational cost, or a general triclinic one. May be
            omitted (and is ignored) if no direction is periodic.
        safe_eval_distance: A value for which ``kernel_fn`` evaluates to a
            result that is not NaN or infinite. Apart from this, it can be
            arbitrary. Used only internally, the exact value has no further
            consequence.

    Returns:
        A function that takes arrays of particle positions and charges,
        and the unit cell, and additionally a neighbor list and pairwise
        weights, as arguments and computes the energy for the whole system
        of particles.
    """
    pbc = onp.asarray(pbc)
    displacement_fn = _concretize_displacement_fn(pbc, cell_mode)

    def compute_energy(
        positions: ArrayLike,
        charges: ArrayLike,
        neighborlist: tuple[ArrayLike, ArrayLike],
        weights: ArrayLike,
        cell: ArrayLike = None,
    ) -> Array:
        """Evaluate pair potential over an entire system, using neighbor list.

        Args:
            positions: Array of positions, shape `(n_particles, n_dim)`.
            charges: Array of charges, shape `(n_particles,)`.
            neighborlist: Tuple of two 1-d integer arrays of the same shape.
                For example, `([0, ..., 26, ...], [91, ..., 5, ...])` would
                mean that particle `91` is a neighbor of particle `0`,
                and particle `5` is a neighbor of particle `26`. Entries `>=
                n_particles` are considered placeholder pairs and do not
                contribute to the energy. They can be used to satisfy the
                static shape requirement in jit-compiled functions.
            weights: Either a scalar or an array of the same shape as each
                component array of `neighbor_list`, representing an extra
                multiplicative weight to be applied to every pairwise energy
                contribution. This can be used to correct for how different
                neighbor list formats may differently handle duplicate
                particle pairs, or even to include something like fudge
                factors for close-together atoms.
            cell: Array representing unit cell, shape `(n_dim, n_dim)`.

        Returns:
            Total system energy.
        """
        # TODO: Support matrix neighbor list format? (would be required for
        #  evaluating the electrostatic potential). Docstring and signature
        #  would need to be adapted.
        # TODO: test calling without `cell` argument in non-periodic case
        if pbc.any():
            if cell is None:
                raise ValueError(
                    "If at least one direction is periodic, "
                    "the cell argument is required."
                )
        (i, j) = neighborlist
        metric_fn = partial(space.metric(displacement_fn), cell=cell)
        mapped_metric_fn = space.map_bond(metric_fn)
        dr_ij = mapped_metric_fn(positions[i], positions[j])
        n_particles = positions.shape[0]
        is_not_placeholder = jnp.logical_and(i < n_particles, j < n_particles)
        # Set distances of placeholder pairs to a value at which the potential
        # can be safely evaluated
        dr_ij = jnp.where(is_not_placeholder, dr_ij, safe_eval_distance)
        qi_qj = charges[i] * charges[j]
        return jnp.where(
            is_not_placeholder, weights * qi_qj * kernel_fn(dr_ij), 0.0
        ).sum()

    return compute_energy


def make_compute_u_zero(
    kernel_fns: Sequence[KernelFn],
    pair_map_fn: Callable[
        [KernelFn],
        Callable[[ArrayLike, ArrayLike, ...], Array],
    ],
) -> Callable[[ArrayLike, ArrayLike, ...], Array]:
    """Create a function that computes the MSM short-range energy contribution.

    The precise quantity being computed is

    .. math::

        U^0 =   \\frac{1}{2} \\sum_i \\sum_{j \\neq i} q_i q_j k_0(r_{ij})
              - \\frac{1}{2} \\sum_{l=1}^L \\sum_i q_i^2 k_{l}(r)\\big\\rvert_{r=0} \, ,

    which consists of the pair interaction term for level zero, and a
    correction term for self-interaction at the higher levels.

    # TODO: Reference to paper?

    This function is a high-level wrapper that constructs the evaluation
    function for :math:`U^0` from two ingredients: The kernel functions
    :math:`k_l(r)` at all levels, and a function that takes care of pair
    distance computation (this includes accounting for periodic boundary
    conditions) and evaluation of such a kernel function over all pairs of
    particles of a charged system.

    Args:
        kernel_fns: List of functions of a single scalar distance argument,
            one for each MSM level, corresponding to the different partial
            kernels into which the full interaction kernel is split.
            They are expected to be natively broadcastable over array inputs.
        pair_map_fn: A function of a single argument that transforms a
            distance-dependent interaction kernel into a pairwise evaluation
            function that acts across a system of charged particles.

            - The input to ``pair_map_fn`` should be a single-argument
              function of a scalar distance argument.

            - The return value of ``pair_map_fn`` should be a function that
              computes the first term in the formula above.
              It takes two arrays (positions of shape `(n_particles, n_dim)`,
              and charges of shape `(n_particles,)`), plus optionally
              additional keyword arguments. Natural use cases for parameters
              passed through keyword arguments would be a unit cell in
              systems with periodicity, or a neighbor list.

            .. note::
               The way that periodic boundary conditions are handled
               is by an appropriate definition of ``pair_map_fn``.

            ``pair_map_fn`` gets applied to the zeroth element of the
            ``kernel_fns`` argument:
            ``compute_pair_term = pair_map_fn(kernel_fns[0])``.

            The most convenient way to obtain a ``pair_map_fn`` with
            appropriate signature is by closing
            :func:`make_eval_pair_pot` or
            :func:`make_eval_pair_pot_neighborlist` over their extra
            arguments, e.g. ``pair_map_fn = functools.partial(
            make_eval_pair_pot, pbc=(True, True, False), cell_mode='ortho')``.

    Returns:
        A function with the same signature as the one returned by
        ``pair_map_fn(kernel_fns[0])``, that computes :math:`U^0` from
        positions, charges, and optional additional keyword arguments.
    """
    compute_pair_term = pair_map_fn(kernel_fns[0])
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    def compute_u_zero(
        positions: ArrayLike,
        charges: ArrayLike,
        **kwargs: ...,
    ) -> Array:
        """Compute the short-range energy contribution :math:`U^0` of the MSM.

        Args:
            positions: Array of positions, shape `(n_particles, n_dim)`.
            charges: Array of charges, shape `(n_particles,)`.
            **kwargs: Optional additional keyword arguments passed to the
                function returned by ``pair_map_fn``. Natural use cases
                would be a unit cell or a neighbor list.

        Returns:
            The short-range energy contribution :math:`U^0`.
        """
        pair_term = compute_pair_term(positions, charges, **kwargs)
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_u_zero
