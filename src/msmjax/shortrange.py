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
import itertools
from functools import partial
from typing import Callable, List, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jaxborlist.neighbor_list import compute_pairwise_deltas
from tqdm import tqdm

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel
from msmjax.utils import _sqrt


def gen_supercell(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    supercell_diag: Union[int, Sequence[int]],
):
    """Adapted from NeuralIL

    TODO: proper attribution
    """
    n_particles, n_dim = positions.shape
    if onp.ndim(supercell_diag) == 0:
        supercell_diag = (supercell_diag,) * n_dim
    M = onp.prod(supercell_diag)
    tile_positions = jnp.tile(positions, (M, 1))
    super_charges = jnp.tile(charges, M)
    grid = jnp.indices(supercell_diag).reshape(n_dim, -1).T
    translations = jnp.dot(grid, cell)
    tile_translations = jnp.repeat(translations, n_particles, axis=0)
    super_positions = tile_positions + tile_translations
    super_cell = cell * jnp.array(supercell_diag)[:, jnp.newaxis]
    return super_positions, super_charges, super_cell


def compute_distance_vectors(positions, cell, pair_indices, pbc):
    deltas = positions[pair_indices[1]] - positions[pair_indices[0]]
    scaled = deltas @ jnp.linalg.pinv(cell)
    return jnp.where(pbc, (scaled - jnp.rint(scaled)) @ cell, deltas)


def make_pair_term_fn(
    kernel_fn: Callable,
    pbc: npt.ArrayLike,
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    # TODO: unit test jitting
    pbc = onp.asarray(pbc)
    if onp.logical_and(~pbc, onp.asarray(supercell_diag) != 1).any():
        raise ValueError(
            "`supercell_diag` must be equal to one along non-periodic axes"
        )
    # TODO: check supercell_diag >= 1?
    _compute_distance_vectors = partial(compute_distance_vectors, pbc=pbc)

    def compute_pair_term(positions, charges, cell):
        super_positions, super_charges, super_cell = gen_supercell(
            positions=positions,
            charges=charges,
            cell=cell,
            supercell_diag=supercell_diag,
        )
        n_centers = positions.shape[0]
        n_total = super_positions.shape[0]  # TODO: from external constant?
        indices_trivial_all_pairs = onp.where(
            onp.arange(n_centers)[:, onp.newaxis] < onp.arange(n_total)
        )
        (i, j) = indices_trivial_all_pairs
        weights_pairs = onp.where(j < n_centers, 1.0, 0.5)
        dR_ij = _compute_distance_vectors(
            positions=super_positions,
            cell=super_cell,
            pair_indices=(i, j),
        )
        dr_ij_2 = (dR_ij * dR_ij).sum(axis=1)
        dr_ij = _sqrt(dr_ij_2)
        qi_qj = super_charges[i] * super_charges[j]

        return (weights_pairs * qi_qj * kernel_fn(dr_ij)).sum()

    return compute_pair_term


def make_pair_term_fn_with_neighbor_list(
    kernel_fn: Callable,
    pbc: npt.ArrayLike,
):
    pbc = onp.asarray(pbc)
    _compute_distance_vectors = partial(compute_distance_vectors, pbc=pbc)

    # TODO: add `pair_weights` parameter (name of parameter?)
    def compute_pair_term(positions, charges, cell, neighbor_list, weights):
        (i, j) = neighbor_list
        dR_ij = _compute_distance_vectors(
            positions=positions,
            pair_indices=(i, j),
            cell=cell,
        )
        dr_ij_2 = (dR_ij * dR_ij).sum(axis=1)
        dr_ij = _sqrt(dr_ij_2)
        n_particles = positions.shape[0]
        is_not_placeholder = jnp.logical_and(i < n_particles, j < n_particles)
        # TODO: Should this "safe distance" be a function argument?
        # Set distances of placeholder pairs to a value at which the potential
        # can be safely evaluated
        dr_ij = jnp.where(is_not_placeholder, dr_ij, 1.0)
        qi_qj = charges[i] * charges[j]
        return jnp.where(
            is_not_placeholder, weights * qi_qj * kernel_fn(dr_ij), 0.0
        ).sum()

    return compute_pair_term


def make_compute_U0(
    kernel_fns: List[Callable],
    pbc: npt.ArrayLike,
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    # TODO: add tests for this
    compute_pair_term = make_pair_term_fn(
        kernel_fn=kernel_fns[0], pbc=pbc, supercell_diag=supercell_diag
    )
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    def compute_U0(positions, charges, cell):
        pair_term = compute_pair_term(
            positions=positions,
            charges=charges,
            cell=cell,
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_U0


def make_compute_U0_with_neighbor_list(
    kernel_fns: List[Callable],
    pbc: npt.ArrayLike,
):
    # TODO: add tests for this
    compute_pair_term = make_pair_term_fn_with_neighbor_list(
        kernel_fn=kernel_fns[0], pbc=pbc
    )
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    # TODO: add `pair_weights` parameter (name of parameter?)
    def compute_U0(positions, charges, cell, neighbor_list):
        pair_term = compute_pair_term(
            positions=positions,
            charges=charges,
            cell=cell,
            indices_pairs=neighbor_list,
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_U0
