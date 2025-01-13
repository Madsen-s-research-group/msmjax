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
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike
from jax_md import space  # TODO: copy to standalone module instead of import
from jax_md.util import (  # # TODO: copy to standalone module instead of import
    f32,
)

from msmjax.utils import _divide_zero_safe


def _gen_supercell(
    positions: ArrayLike,
    charges: ArrayLike,
    cell: ArrayLike,
    supercell_diag: Sequence[int],
):
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


def _generalized_diagonal_mask(X: ArrayLike) -> Array:
    """Set the diagonal of a, possibly wider than tall, matrix to zero.

    Adapted from JAX-MD. # TODO: JAX-MD attribution

    Args:
        X: Original matrix.

    Returns:
        The matrix with diagonal set to zero.
    """
    if len(X.shape) != 2:
        raise ValueError("Only two-dimensional arrays are supported.")
    M, N = X.shape
    if M > N:
        raise ValueError(
            "Input array must be either square, or wider than tall."
        )
    X = jnp.nan_to_num(X)
    mask = f32(1.0) - jnp.eye(M, dtype=X.dtype)
    mask = jnp.pad(
        mask,
        pad_width=((0, 0), (0, N - M)),
        mode="constant",
        constant_values=1,
    )
    return mask * X


def _displacement_free(R_1, R_2):
    # TODO: unit test this on its own?
    return R_1 - R_2


def _displacement_ortho(R_1, R_2, side_lengths):
    # TODO: unit test this on its own?
    delta = R_1 - R_2
    return (
        delta
        - jnp.round(_divide_zero_safe(delta, side_lengths)) * side_lengths
    )


def _displacement_general(R_1, R_2, cell):
    # TODO: unit test this on its own?
    dR = R_1 - R_2
    inv_cell = jnp.linalg.pinv(cell)
    R_1_transf = R_1 @ inv_cell
    R_2_transf = R_2 @ inv_cell
    dR_transformed = R_1_transf - R_2_transf
    return dR - jnp.round(dR_transformed) @ cell


def _concretize_displacement_fn(
    pbc: Sequence[bool],
    cell_mode: Optional[
        Literal["ortho", "general"]
    ] = None,  # TODO: define the allowed values globally
):
    """Select/construct displacement fn based on PBCs, unit cell constraints.

    Wraps lower-level displacement functions and transforms them into ones
    with a choice of periodic boundary conditions built-in already, and with
    a consistent signature.

    Args:
        pbc: One boolean per direction signaling periodicity.
        cell_mode: # TODO

    Returns:
        A function of two position vector arguments and, (optionally, depending
        on PBCs) a unit cell, that computes the distance between them.
    """
    # TODO: type hint for cell_mode (in all places where it's used)
    if jnp.any(pbc) and cell_mode is None:
        # TODO: write test for this check
        raise ValueError(
            "If at least one direction is periodic, "
            "you must specify cell_mode."
        )

    if not jnp.any(pbc):

        def displacement_fn(R_1, R_2, cell=None):
            return _displacement_free(R_1, R_2)

        return displacement_fn

    if cell_mode == "ortho":

        def displacement_fn(R_1, R_2, cell):
            side_lengths_processed_for_pbc = jnp.diag(cell) * pbc
            return _displacement_ortho(
                R_1, R_2, side_lengths_processed_for_pbc
            )

        return displacement_fn

    elif cell_mode == "general":

        def displacement_fn(R_1, R_2, cell):
            cell_processed_for_pbc = cell * pbc[:, jnp.newaxis]
            return _displacement_general(R_1, R_2, cell_processed_for_pbc)

        return displacement_fn

    else:
        raise ValueError("Invalid cell_mode.")


def make_pair_term_fn(
    kernel_fn: Callable,
    pbc: Sequence[bool],
    cell_mode: Optional[
        Literal["ortho", "general"]
    ] = None,  # TODO: define the allowed values globally
    supercell_diag: Optional[Sequence[int]] = None,
):
    """Transform interaction kernel into function acting on a particle system.

    In other words, given a distance-dependent interaction kernel :math:`k(r)`,
    construct another function that computes the total system energy,
    :math:`\\frac{1}{2} \\sum_i \\sum_{j \\neq i} q_i q_j k(r_{ij})`, by mapping
    :math:`k(r)` over all particle pairs.

    # TODO: Mention MIC limitations here? (maybe format as warning)

    Args:
        kernel_fn: A function of a single scalar distance argument,
            corresponding to :math:`k(r)` in the above formula.
        pbc: One boolean per direction signaling periodicity.
        cell_mode: May be omitted (and is ignored) if no direction is periodic. # TODO: finish
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
    # TODO: function name maybe not ideal
    pbc = onp.asarray(pbc)
    if supercell_diag is None:
        supercell_diag = onp.ones_like(pbc, dtype=int)
    if onp.logical_and(~pbc, onp.asarray(supercell_diag) != 1).any():
        raise ValueError(
            "`supercell_diag` must be equal to one along non-periodic axes"
        )
    # TODO: check supercell_diag >= 1?

    displacement_fn = _concretize_displacement_fn(pbc, cell_mode)

    def compute_pair_term(
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
            0.5 * (_generalized_diagonal_mask(qi_qj * kernel_fn(dr_ij))).sum()
        )

    return compute_pair_term


def make_pair_term_fn_with_neighbor_list(
    kernel_fn: Callable,
    pbc: Sequence[bool],
    cell_mode: Optional[
        Literal["ortho", "general"]
    ] = None,  # TODO: define the allowed values globally
    safe_eval_distance: float = 1.0,  # TODO: test? (how?)
):
    """Transform interaction kernel into function acting on a particle system.

    Like :func:`make_pair_term_fn`, but with a neighbor list.

    Args:
        kernel_fn: A function of a single scalar distance argument,
            see documentation of :func:`make_pair_term_fn`.
        pbc: One boolean per direction signaling periodicity.
        cell_mode: # TODO
        safe_eval_distance: A value for which ``kernel_fn`` evaluates to a
            result that is not `nan` or `inf`. Apart from this, it can be
            arbitrary and its exact value is of no consequence. Used
            internally in safely ignoring placeholder pairs contained in the
            neighbor list in a jit- and autodiff-compatible way.

    Returns:
        A function that takes arrays of particle positions and charges,
        and the unit cell, and additionally a neighbor list and pairwise
        weights, as arguments and computes the energy for the whole system
        of particles.
    """
    # TODO: function name maybe not ideal
    pbc = onp.asarray(pbc)
    displacement_fn = _concretize_displacement_fn(pbc, cell_mode)

    def compute_pair_term(
        positions: ArrayLike,
        charges: ArrayLike,
        cell: ArrayLike,
        neighbor_list: Tuple[ArrayLike, ArrayLike],
        weights: ArrayLike,
    ) -> Array:
        """Evaluate pair potential over an entire system, using neighbor list.

        Args:
            positions: Array of positions, shape `(n_particles, n_dim)`.
            charges: Array of charges, shape `(n_particles,)`.
            cell: Array representing unit cell, shape `(n_dim, n_dim)`.
            neighbor_list: Tuple of two 1-d integer arrays of the same shape.
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

        Returns:
            Total system energy.
        """
        # TODO: Support matrix neighbor list format? (would be required for
        #  evaluating the electrostatic potential). Docstring and signature
        #  would need to be adapted.
        (i, j) = neighbor_list
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

    return compute_pair_term


def make_compute_U0(
    kernel_fns: List[Callable],
    pair_map_fn: Callable[[Callable], Callable],
):
    """Create a function that computes the MSM short-range energy contribution.

    # TODO: formula?

    .. code-block:: python

        compute_pair_term = pair_map_fn(kernel_fns[0])

    Args:
        kernel_fns: List of functions of a single scalar distance argument,
            one for each MSM level, corresponding to the different partial
            kernels into which the full interaction kernel is split.
        pair_map_fn: A function of one argument that transforms an interaction
            kernel function like :math:`k(r)` into a function that evaluates it
            pairwise for a whole system of charged particles.  # TODO
            This gets applied to the 0-th element of ``kernel_fns``.

    Returns:
        A function that takes arrays of particle positions and charges,
        and the unit cell, as arguments and computes :math:`U^0` for the
        whole system of particles.
    """
    # TODO: lowercase function name?
    compute_pair_term = pair_map_fn(kernel_fns[0])
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    def compute_U0(positions, charges, cell, **kwargs):
        """Compute the short-range energy contribution :math:`U^0` of the MSM.

        Args:
            positions: Array of positions, shape `(n_particles, n_dim)`.
            charges: Array of charges, shape `(n_particles,)`.
            cell: Array representing unit cell, shape `(n_dim, n_dim)`.
            **kwargs: Optional additional keyword arguments passed to the
                function returned by ``pair_map_fn``. A natural use case would
                be a neighbor list.

        Returns:

        """
        pair_term = compute_pair_term(
            positions=positions, charges=charges, cell=cell, **kwargs
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_U0
