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

import os

# TODO: remove once all actual code-running lines have been moved to tests
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

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


def _evaluate_pairs(
    positions,
    charges,
    cell,
    indices_pairs,
    kernel_fn: Callable,
    pbc: jax.Array,
):
    def apply_mic(deltas, cell):
        scaled = deltas @ jnp.linalg.pinv(cell)
        return jnp.where(pbc, (scaled - jnp.rint(scaled)) @ cell, deltas)

    n_particles = positions.shape[0]
    is_not_placeholder = jnp.logical_and(
        indices_pairs[0] < n_particles, indices_pairs[1] < n_particles
    )
    dR = positions[indices_pairs[1]] - positions[indices_pairs[0]]
    dR = apply_mic(dR, cell)
    dr_2 = (dR * dR).sum(axis=1)
    dr = _sqrt(dr_2)
    # TODO: Should this "safe distance" be a function argument?
    # Set distances of placeholder pairs to a value at which the potential
    # can be safely evaluated
    dr = jnp.where(is_not_placeholder, dr, 1.0)
    qi_qj = charges[indices_pairs[0]] * charges[indices_pairs[1]]

    return jnp.where(is_not_placeholder, qi_qj * kernel_fn(dr), 0.0).sum()


def make_pair_term_fn(
    kernel_fn: Callable,
    pbc: npt.ArrayLike,
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    pbc = onp.asarray(pbc)
    _compute_pair_term = partial(_evaluate_pairs, kernel_fn=kernel_fn, pbc=pbc)

    def compute_pair_term(positions, charges, cell):
        super_positions, super_charges, super_cell = gen_supercell(
            positions=positions,
            charges=charges,
            cell=cell,
            supercell_diag=supercell_diag,
        )
        n_centers = positions.shape[0]
        n_total = super_positions.shape[0]  # TODO: from external constant?
        # TODO: Should the construction of these pair indices be put into a
        #  separate function (for isolated testing)?
        indices_trivial_all_pairs = onp.where(
            onp.arange(n_centers)[:, onp.newaxis] < onp.arange(n_total)
        )
        pair_term = _compute_pair_term(
            positions=super_positions,
            charges=super_charges,
            cell=super_cell,
            indices_pairs=indices_trivial_all_pairs,
        )
        return pair_term

    return compute_pair_term


def make_pair_term_fn_with_neighbor_list(
    kernel_fn: Callable,
    pbc: npt.ArrayLike,
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    pbc = onp.asarray(pbc)
    _compute_pair_term = partial(_evaluate_pairs, kernel_fn=kernel_fn, pbc=pbc)

    def compute_pair_term(positions, charges, cell, neighbor_list):
        super_positions, super_charges, super_cell = gen_supercell(
            positions=positions,
            charges=charges,
            cell=cell,
            supercell_diag=supercell_diag,
        )
        pair_term = _compute_pair_term(
            positions=super_positions,
            charges=super_charges,
            cell=super_cell,
            indices_pairs=neighbor_list,
        )
        return pair_term

    return compute_pair_term


def make_compute_U0(
    kernel_fns: List[Callable],
    pbc: npt.ArrayLike,
    supercell_diag: Union[int, Sequence[int]] = 1,
):
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
    supercell_diag: Union[int, Sequence[int]] = 1,
):
    compute_pair_term = make_pair_term_fn_with_neighbor_list(
        kernel_fn=kernel_fns[0], pbc=pbc, supercell_diag=supercell_diag
    )
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

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


if __name__ == "__main__":
    # Structure settings
    # BOX_LENGTHS = jnp.array([10.0, 12.0, 17.5, 20.0])
    BOX_LENGTHS = jnp.array([10.0, 12.0, 17.5])
    # BOX_LENGTHS = jnp.array([10.0, 12.0])
    # BOX_LENGTHS = jnp.array([10.0])
    PERIODIC = True
    N_PARTICLES = 50

    # MSM settings
    LEVEL_ZERO_CUTOFF = 3.5
    MAX_GRIDLEVEL = 4
    P = 4

    n_dim = len(BOX_LENGTHS)
    pbcs = [PERIODIC] * n_dim

    rng = onp.random.default_rng(58347)
    pos = rng.uniform(
        low=[0.0] * n_dim, high=BOX_LENGTHS, size=(N_PARTICLES, n_dim)
    )
    chg = rng.uniform(low=-1.0, high=1.0, size=N_PARTICLES)
    pos = jnp.array(pos)
    chg = jnp.array(chg)

    kernels = split_one_over_r_kernel(
        max_level=MAX_GRIDLEVEL,
        level_zero_cutoff=LEVEL_ZERO_CUTOFF,
        softening_function=SofteningFunctionOneOverR(P),
    )

    neighbor_fun, energy_fun = make_compute_U_zero_with_neighborlist(
        kernels=kernels,
        cutoff=LEVEL_ZERO_CUTOFF,
        box_lengths=BOX_LENGTHS,
        pbcs=pbcs,
    )
    nbl_allocate_fun = neighbor_fun.allocate
    nbl_update_fun = neighbor_fun.update
    energy_fun = jax.jit(energy_fun)
    neighborlist = nbl_allocate_fun(pos)
    e_neighborlist = energy_fun(pos, chg, neighborlist.idx)
    print(e_neighborlist)

    @jax.jit
    def wrapper_energy_neighborlist(positions, charges):
        updated_neighborlist = nbl_update_fun(positions, neighborlist)
        return energy_fun(positions, charges, updated_neighborlist.idx)

    @jax.jit
    def wrapper_forces_neighborlist(positions, charges):
        return -jax.grad(wrapper_energy_neighborlist, argnums=0)(
            positions, charges
        )

    e_from_wrapper = wrapper_energy_neighborlist(pos, chg)
    f_from_wrapper = wrapper_forces_neighborlist(pos, chg)

    print(e_from_wrapper)

    _, force_fun = make_compute_f_zero_with_neighborlist(
        kernels=kernels,
        cutoff=LEVEL_ZERO_CUTOFF,
        box_lengths=BOX_LENGTHS,
        pbcs=pbcs,
    )
    force_fun = jax.jit(force_fun)
    f_neighborlist = force_fun(pos, chg, neighborlist.idx)

    (_, energy_and_force_fun,) = make_compute_U_and_f_zero_with_neighborlist(
        kernels=kernels,
        cutoff=LEVEL_ZERO_CUTOFF,
        box_lengths=BOX_LENGTHS,
        pbcs=pbcs,
    )
    energy_and_force_fun = jax.jit(energy_and_force_fun)
    e_nbl_comb, f_nbl_comb = energy_and_force_fun(pos, chg, neighborlist.idx)
    print(e_nbl_comb)
    print(f_nbl_comb)

    def calculate_direct_energy_reference(
        positions, charges, cutoff, mic=False, box_sizes=None
    ):
        shortrange_kernel = kernels[0]
        sum_of_higher_kernels_at_zero = jnp.sum(
            jnp.asarray([k(0.0) for k in kernels[1:]])
        )

        if mic:

            def apply_boundary_conditions(R):
                scaled_R = R / box_sizes
                scaled_R -= onp.rint(scaled_R)
                return scaled_R * box_sizes

        else:
            apply_boundary_conditions = lambda x: x

        pair_term = 0.0
        for i in tqdm(range(positions.shape[0])):
            for j in range(i):
                R_ij = positions[i] - positions[j]
                R_ij = apply_boundary_conditions(R_ij)
                r_ij_2 = (R_ij * R_ij).sum()
                if r_ij_2 <= cutoff**2:
                    r_ij = onp.sqrt(r_ij_2)
                    pair_term += (
                        charges[i] * charges[j] * shortrange_kernel(r_ij)
                    )

        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )

        return pair_term - self_interaction_term

    e_ref = calculate_direct_energy_reference(
        pos, chg, LEVEL_ZERO_CUTOFF, mic=PERIODIC, box_sizes=BOX_LENGTHS
    )
    print(e_ref)
    assert jnp.isclose(e_neighborlist, e_ref)
    assert jnp.isclose(e_nbl_comb, e_ref)

    def calculate_direct_forces_reference(
        positions, charges, cutoff, mic=False, box_sizes=None
    ):
        shortrange_kernel = kernels[0]
        k_0_prime = jax.grad(shortrange_kernel)

        if mic:

            def apply_boundary_conditions(R):
                scaled_R = R / box_sizes
                scaled_R -= onp.rint(scaled_R)
                return scaled_R * box_sizes

        else:
            apply_boundary_conditions = lambda x: x

        n_particles = positions.shape[0]
        n_dim = positions.shape[1]
        forces = []
        for i in tqdm(range(n_particles)):
            f_i = jnp.zeros(n_dim)
            for j in itertools.chain(range(i), range(i + 1, n_particles)):
                R_ij = positions[i] - positions[j]
                R_ij = apply_boundary_conditions(R_ij)
                r_ij_2 = (R_ij * R_ij).sum()
                if r_ij_2 <= cutoff**2:
                    r_ij = onp.sqrt(r_ij_2)
                    f_i -= (
                        charges[i] * charges[j] * k_0_prime(r_ij) * R_ij / r_ij
                    )
            forces.append(f_i)

        return jnp.array(forces)

    f_ref = calculate_direct_forces_reference(
        pos, chg, LEVEL_ZERO_CUTOFF, mic=PERIODIC, box_sizes=BOX_LENGTHS
    )
    assert jnp.allclose(f_ref, f_from_wrapper)
    assert jnp.allclose(f_ref, f_neighborlist)
    assert jnp.allclose(f_ref, f_nbl_comb)
