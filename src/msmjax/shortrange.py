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
from jax_md import space  # TODO
from jax_md.util import f32  # TODO

from msmjax.utils import _sqrt


def gen_supercell(
    positions: jax.Array,
    charges: jax.Array,
    cell: jax.Array,
    supercell_diag: Union[
        int, Sequence[int]
    ],  # TODO: does the type of `supercell_diag` need to be more specific?
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


def _nonperiodic_displacement(R_1, R_2, cell):
    return R_1 - R_2


def _periodic_displacement_general(R_1, R_2, cell):
    # Transpose the cell to make it compatible with JAX-MD
    cell = cell.T
    # TODO: change inv to pinv in inverse function?
    inv_cell = space.inverse(cell)
    R_1 = space.transform(inv_cell, R_1)
    R_2 = space.transform(inv_cell, R_2)
    dR = space.periodic_displacement(
        f32(1.0), space.pairwise_displacement(R_1, R_2)
    )
    dR = space.transform(cell, dR)
    return dR


def _periodic_displacement_ortho(R_1, R_2, cell):
    return _periodic_displacement_general(R_1, R_2, cell=jnp.diag(cell))


def select_displacement_fn(pbc, cell_type) -> Callable:
    if onp.all(~pbc):
        return _nonperiodic_displacement
    else:
        if cell_type == "ortho":
            periodic_disp = _periodic_displacement_ortho
        elif cell_type == "general":
            periodic_disp = _periodic_displacement_general
        else:
            # TODO: better error message
            raise ValueError("Illegal value for `cell_type`")
    if onp.all(pbc):
        return periodic_disp
    else:
        return lambda R_1, R_2, cell: jnp.where(
            pbc,
            periodic_disp(R_1, R_2, cell=cell),
            _nonperiodic_displacement(R_1, R_2, cell=cell),
        )


def _generalized_diagonal_mask(X):
    """Set the diagonal of a matrix to zero that may be wider than tall"""
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


def make_pair_term_fn(
    kernel_fn: Callable,
    pbc: npt.ArrayLike,
    # TODO: Default value for `cell_type`: Is `None` okay?
    cell_type: Optional[Literal["ortho", "general"]] = None,
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    # TODO: unit test jitting
    pbc = onp.asarray(pbc)
    supercell_diag = onp.asarray(supercell_diag)
    if onp.logical_and(~pbc, onp.asarray(supercell_diag) != 1).any():
        raise ValueError(
            "`supercell_diag` must be equal to one along non-periodic axes"
        )
    # TODO: check supercell_diag >= 1?

    displacement_fn = select_displacement_fn(pbc, cell_type)

    def compute_pair_term(positions, charges, cell):
        super_positions, super_charges, super_cell = gen_supercell(
            positions=positions,
            charges=charges,
            cell=cell,
            supercell_diag=supercell_diag,
        )
        metric_fn = partial(space.metric(displacement_fn), cell=super_cell)
        mapped_metric_fn = space.map_product(metric_fn)
        # TODO: order of arguments to metric?
        dr_ij = mapped_metric_fn(super_positions, positions)
        qi_qj = charges[:, jnp.newaxis] * super_charges
        return (
            0.5 * (_generalized_diagonal_mask(qi_qj * kernel_fn(dr_ij))).sum()
        )

    return compute_pair_term


def make_pair_term_fn_with_neighbor_list(
    kernel_fn: Callable,
    pbc: npt.ArrayLike,
    cell_type,  # TODO: type hint, default value?
):
    pbc = onp.asarray(pbc)
    displacement_fn = select_displacement_fn(pbc, cell_type)

    def compute_pair_term(positions, charges, cell, neighbor_list, weights):
        (i, j) = neighbor_list
        metric_fn = partial(space.metric(displacement_fn), cell=cell)
        mapped_metric_fn = space.map_bond(metric_fn)
        dr_ij = mapped_metric_fn(positions[i], positions[j])
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
    cell_type,  # TODO: type hint, default value?
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    compute_pair_term = make_pair_term_fn(
        kernel_fn=kernel_fns[0],
        pbc=pbc,
        cell_type=cell_type,
        supercell_diag=supercell_diag,
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
    cell_type,  # TODO: type hint, default value?
):
    # TODO: add tests for this
    # TODO: Using a neighbor list together with `supercell_diag` could get a
    #  bit complicated, so this function currently doesn't support
    #  `supercell_diag`, but it could still be useful.
    compute_pair_term = make_pair_term_fn_with_neighbor_list(
        kernel_fn=kernel_fns[0], pbc=pbc, cell_type=cell_type
    )
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    # TODO: add `pair_weights` parameter (name of parameter?)
    def compute_U0(positions, charges, cell, neighbor_list, weights):
        pair_term = compute_pair_term(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_list=neighbor_list,
            weights=weights,
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_U0
