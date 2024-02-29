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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import itertools
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jax_md import partition, space
from tqdm import tqdm

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel


def make_evaluate_shortrange_with_neighbor_list(
    kernels: List[Callable],
    cutoff: float,
    box: npt.ArrayLike,
    pbcs: npt.ArrayLike,
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, Callable]:
    box = jnp.asarray(box)
    pbcs = jnp.asarray(pbcs)

    if not (jnp.all(pbcs) or jnp.all(~pbcs)):
        raise ValueError("Mixed boundary conditions currently not supported.")
    periodic = pbcs[0]
    if periodic and cutoff > 0.5 * min(box):
        raise ValueError("Cutoff must not exceed half the shortest box length")

    shortrange_kernel = kernels[0]
    sum_of_higher_kernels_at_zero = jnp.sum(
        jnp.asarray([k(0.0) for k in kernels[1:]])
    )

    if periodic:
        displacement_fn, shift_fn = space.periodic(box)
    else:
        displacement_fn, shift_fn = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn, box, r_cutoff=cutoff, **neighbor_kwargs
    )

    def evaluate_one_row_of_neighborlist(idx_of_row, row, dR, qq):
        nb_particles = dR.shape[0]
        pair_contribs = jnp.where(
            row < nb_particles,
            qq[idx_of_row, row]
            * shortrange_kernel(jnp.linalg.norm(dR[idx_of_row, row], axis=1)),
            0.0,
        )

        return jnp.sum(pair_contribs)

    def calculate_direct_energy_neighborlist(positions, charges, neighborlist):
        dR = space.map_product(displacement_fn)(positions, positions)
        qq = charges[:, jnp.newaxis] * charges
        all_indices = jnp.arange(neighborlist.idx.shape[0])
        pair_term = 0.5 * jnp.sum(
            jax.vmap(evaluate_one_row_of_neighborlist, (0, 0, None, None))(
                all_indices, neighborlist.idx, dR, qq
            )
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )

        return pair_term - self_interaction_term

    return neighbor_fn, calculate_direct_energy_neighborlist


if __name__ == "__main__":
    BOX_LENGHTS = jnp.array([10.0, 10.0, 10.0])
    PERIODIC = False
    N_PARTICLES = 25

    LEVEL_ZERO_CUTOFF = 3.5
    MAX_GRIDLEVEL = 4
    P = 4

    n_dim = len(BOX_LENGHTS)
    pbcs = [PERIODIC] * n_dim

    rng = onp.random.default_rng(58347)
    pos = rng.uniform(
        low=[0.0] * n_dim, high=BOX_LENGHTS, size=(N_PARTICLES, n_dim)
    )
    chg = rng.uniform(low=-1.0, high=1.0, size=N_PARTICLES)
    pos = jnp.array(pos)
    chg = jnp.array(chg)

    kernels = split_one_over_r_kernel(
        max_level=MAX_GRIDLEVEL,
        level_zero_cutoff=LEVEL_ZERO_CUTOFF,
        softening_function=SofteningFunctionOneOverR(P),
    )

    neighbor_fun, energy_fun = make_evaluate_shortrange_with_neighbor_list(
        kernels=kernels,
        cutoff=LEVEL_ZERO_CUTOFF,
        box=BOX_LENGHTS,
        pbcs=pbcs,
    )
    nbl_allocate_fun = neighbor_fun.allocate
    nbl_update_fun = neighbor_fun.update
    neighborlist = nbl_allocate_fun(pos)
    e_neighborlist = energy_fun(pos, chg, neighborlist)
    print(e_neighborlist)

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
                # TODO: check inside cutoff here?
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

    def calculate_direct_energy_reference_periodic(
        positions, charges, cell, cutoff
    ):
        shortrange_kernel = jax.jit(kernels[0])
        sum_of_higher_kernels_at_zero = jnp.sum(
            jnp.asarray([k(0.0) for k in kernels[1:]])
        )

        n_dim = positions.shape[1]
        n_cells_inside_cutoff = [int(onp.ceil(cutoff / l)) for l in cell]
        cellshifts_1d = [
            onp.arange(-n_reps, n_reps + 1) for n_reps in n_cells_inside_cutoff
        ]
        cellshifts_nonzero = [
            onp.array(cs)
            for cs in itertools.product(*cellshifts_1d)
            if cs != (0,) * n_dim
        ]
        positions_extended = [positions] + [
            positions + cs * BOX_LENGHTS for cs in cellshifts_nonzero
        ]
        positions_extended = onp.concatenate(positions_extended)
        charges_extended = onp.tile(charges, len(cellshifts_nonzero) + 1)

        pair_term = 0.0
        for i in tqdm(range(positions.shape[0])):
            for j in itertools.chain(
                range(i), range(i + 1, positions_extended.shape[0])
            ):
                r_ij = onp.linalg.norm(positions[i] - positions_extended[j])
                pair_term += (
                    charges[i] * charges_extended[j] * shortrange_kernel(r_ij)
                )
        pair_term *= 0.5

        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )

        return pair_term - self_interaction_term

    e_ref = calculate_direct_energy_reference(
        pos, chg, LEVEL_ZERO_CUTOFF, mic=PERIODIC, box_sizes=BOX_LENGHTS
    )

    print(e_ref)

    assert jnp.isclose(e_neighborlist, e_ref)
