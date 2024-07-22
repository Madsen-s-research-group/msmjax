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
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jax_md import partition, space
from jaxborlist.neighbor_list import compute_pairwise_deltas
from tqdm import tqdm

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel


def make_eval_all_pairs_neighborlist(
    pair_distance_vector_fun, pair_eval_fun, pair_eval_fun_output_shape
):
    # TODO: docstring
    def process_one_particle_and_neighbors(
        idx_center: int,
        inds_neighbors: jax.Array,
        R_ij: jax.Array,
        charges: jax.Array,
    ) -> jax.Array:
        """Compute energy contribution due to one particle and its neighbors.

        Args:
            idx_center: Index of the central particle.
            inds_neighbors: 1-d array of indices of neighbor particles to be
                included in energy computation, of shape (n_max_neighbors,).
                Index values greater than or equal to the total number of
                particles are interpreted as fill values and ignored.
            R_ij: Array of pairwise distance vectors, of shape
                `(n_particles, n_particles, n_dim)`. They are passed to the
                pair potential as-is, i.e., in the case of periodic boundary
                conditions the minimum image convention must have been
                accounted for beforehand.
            charges: Array of particle charges, of shape `(n_particles,)`.

        Returns:
            Array of length `n_max_neighbors` along the first axis, containing
            the contribution from each neighbor.
        """
        n_particles = R_ij.shape[0]
        is_neighbor = inds_neighbors < n_particles
        neighbor_cond_made_shape_compatible = jnp.tile(
            is_neighbor, (*pair_eval_fun_output_shape, 1)
        ).T
        pair_contribs = jnp.where(
            neighbor_cond_made_shape_compatible,
            jax.vmap(pair_eval_fun, in_axes=(0, None, 0))(
                R_ij[idx_center, inds_neighbors],
                charges[idx_center],
                charges[inds_neighbors],
            ),
            0.0,
        )

        return pair_contribs

    def eval_all_pairs_with_neighborlist(
        positions: jax.Array,
        charges: jax.Array,
        neighbor_indices: jax.Array,
    ) -> jax.Array:
        """Compute the direct energy contribution ($U^0$ in the article).

        Args:
            positions: Particle positions in cartesian coordinates,
                array of shape `(n_particles, n_dim)`.
            charges: Particle charges, array of shape `(n_particles,)`.
            neighbor_indices: Array of shape `(n_particles, n_max_neighbors)`,
                where the `i`-th row contains the indices of the neighbors of
                particle `i`. Index values greater than or equal to the total
                number of particles are interpreted as fill values (padding to
                consistent neighbor list shape plus spare capacity), and
                ignored in the evaluation.  # TODO

        Returns:
            The calculated energy contribution. # TODO
        """
        R_ij = pair_distance_vector_fun(positions)
        all_indices = jnp.arange(neighbor_indices.shape[0])
        return jax.vmap(
            process_one_particle_and_neighbors, (0, 0, None, None)
        )(all_indices, neighbor_indices, R_ij, charges)

    return eval_all_pairs_with_neighborlist


def make_compute_U_zero_with_neighborlist(
    kernels: List[Callable],
    cutoff: float,
    box_lengths: npt.ArrayLike,
    pbcs: npt.ArrayLike,
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, Callable]:
    """Get functions handling neighbor list, and computing direct energy part.

    Args:
        kernels: List of one-argument functions of a scalar argument,
            representing the interaction kernels at all levels.
        cutoff: Cutoff radius of the level-zero kernel (which is evaluated
            directly rather than by interpolation from grids).
        box_lengths: Sequence of side lengths of simulation box, one per
            spatial dimension.
        pbcs: Sequence of boolean values indicating whether the system is
            periodic along the corresponding direction.
        **neighbor_kwargs: Will be passed to `jax_md.partition.neighbor_list`.

    Returns:
        Tuple containing:
            - An instance of `jax_md.partition.NeighborListFns` that in turn
                consists of functions to allocate and update a neighbor list.
            - A function for calculating the direct energy contribution from
                given particle positions, charges, and a neighbor list.

    """
    box_lengths = jnp.asarray(box_lengths)
    pbcs = jnp.asarray(pbcs)

    # TODO: make mixed boundary conditions work
    if not (jnp.all(pbcs) or jnp.all(~pbcs)):
        raise ValueError("Mixed boundary conditions currently not supported.")
    periodic = pbcs[0]
    if periodic and cutoff > 0.5 * min(box_lengths):
        raise ValueError("Cutoff must not exceed half the shortest box length")

    shortrange_kernel = kernels[0]
    sum_of_higher_kernels_at_zero = jnp.sum(
        jnp.asarray([k(0.0) for k in kernels[1:]])
    )

    if periodic:
        displacement_fn, shift_fn = space.periodic(box_lengths)
    else:
        displacement_fn, shift_fn = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn, box_lengths, r_cutoff=cutoff, **neighbor_kwargs
    )

    n_dim = box_lengths.shape[0]

    def compute_pair_distance_vectors(positions):
        # With the extra minus, the element `(i, j)` of the output is equal
        # to (under boundary conditions) `positions[i] - positions[j]`
        return -space.map_product(displacement_fn)(positions, positions)

    def pair_eval_fun(R_ij, qi, qj):
        return qi * qj * shortrange_kernel(jnp.linalg.norm(R_ij))

    eval_shortrange_all_pairs_neighborlist = make_eval_all_pairs_neighborlist(
        pair_distance_vector_fun=compute_pair_distance_vectors,
        pair_eval_fun=pair_eval_fun,
        pair_eval_fun_output_shape=pair_eval_fun(
            jnp.ones(n_dim), 1.0, 1.0
        ).shape,
    )

    def compute_U_zero_with_neighborlist(
        positions: jax.Array,
        charges: jax.Array,
        neighbor_indices: jax.Array,
    ) -> jax.Array:
        """Compute the direct energy contribution ($U^0$ in the article).

        Args:
            positions: Particle positions in cartesian coordinates,
                array of shape `(n_particles, n_dim)`.
            charges: Particle charges, array of shape `(n_particles,)`.
            neighbor_indices: Array of shape `(n_particles, n_max_neighbors)`,
                where the `i`-th row contains the indices of the neighbors of
                particle `i`. Index values greater than or equal to the total
                number of particles are interpreted as fill values (padding to
                consistent neighbor list shape plus spare capacity), and
                ignored in the energy evaluation.

        Returns:
            The calculated energy contribution.
        """
        pair_term = 0.5 * jnp.sum(
            eval_shortrange_all_pairs_neighborlist(
                positions, charges, neighbor_indices
            )
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )

        return pair_term - self_interaction_term

    return neighbor_fn, compute_U_zero_with_neighborlist


def make_compute_f_zero_with_neighborlist(
    kernels: List[Callable],
    cutoff: float,
    box_lengths: npt.ArrayLike,
    pbcs: npt.ArrayLike,
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, Callable]:
    """Get functions handling neighbor list, and computing direct energy part.

    Args:
        kernels: List of one-argument functions of a scalar argument,
            representing the interaction kernels at all levels.
        cutoff: Cutoff radius of the level-zero kernel (which is evaluated
            directly rather than by interpolation from grids).
        box_lengths: Sequence of side lengths of simulation box, one per
            spatial dimension.
        pbcs: Sequence of boolean values indicating whether the system is
            periodic along the corresponding direction.
        **neighbor_kwargs: Will be passed to `jax_md.partition.neighbor_list`.

    Returns:
        Tuple containing:
            - An instance of `jax_md.partition.NeighborListFns` that in turn
                consists of functions to allocate and update a neighbor list.
            - A function for calculating the direct energy contribution from
                given particle positions, charges, and a neighbor list.

    """
    box_lengths = jnp.asarray(box_lengths)
    pbcs = jnp.asarray(pbcs)

    # TODO: make mixed boundary conditions work
    if not (jnp.all(pbcs) or jnp.all(~pbcs)):
        raise ValueError("Mixed boundary conditions currently not supported.")
    periodic = pbcs[0]
    if periodic and cutoff > 0.5 * min(box_lengths):
        raise ValueError("Cutoff must not exceed half the shortest box length")

    shortrange_kernel = kernels[0]
    k_0_prime = jax.grad(shortrange_kernel)
    # sum_of_higher_kernels_at_zero = jnp.sum(
    #     jnp.asarray([k(0.0) for k in kernels[1:]])
    # )

    if periodic:
        displacement_fn, shift_fn = space.periodic(box_lengths)
    else:
        displacement_fn, shift_fn = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn, box_lengths, r_cutoff=cutoff, **neighbor_kwargs
    )

    n_dim = box_lengths.shape[0]

    def compute_pair_distance_vectors(positions):
        # With the extra minus, the element `(i, j)` of the output is equal
        # to (under boundary conditions) `positions[i] - positions[j]`
        return -space.map_product(displacement_fn)(positions, positions)

    def pair_eval_fun(R_ij, q_i, q_j):
        r_ij = jnp.linalg.norm(R_ij)
        return -q_i * q_j * k_0_prime(r_ij) * R_ij / r_ij

    eval_fun = make_eval_all_pairs_neighborlist(
        pair_distance_vector_fun=compute_pair_distance_vectors,
        pair_eval_fun=pair_eval_fun,
        pair_eval_fun_output_shape=pair_eval_fun(
            jnp.ones(n_dim), 1.0, 1.0
        ).shape,
    )

    def compute_f_zero_with_neighborlist(
        positions: jax.Array,
        charges: jax.Array,
        neighbor_indices: jax.Array,
    ) -> jax.Array:
        """Compute the direct energy contribution ($U^0$ in the article).

        Args:
            positions: Particle positions in cartesian coordinates,
                array of shape `(n_particles, n_dim)`.
            charges: Particle charges, array of shape `(n_particles,)`.
            neighbor_indices: Array of shape `(n_particles, n_max_neighbors)`,
                where the `i`-th row contains the indices of the neighbors of
                particle `i`. Index values greater than or equal to the total
                number of particles are interpreted as fill values (padding to
                consistent neighbor list shape plus spare capacity), and
                ignored in the energy evaluation.

        Returns:
            The calculated energy contribution.
        """
        forces = jnp.sum(
            eval_fun(positions, charges, neighbor_indices),
            axis=1,
        )

        return forces

    return neighbor_fn, compute_f_zero_with_neighborlist


def make_compute_U_and_f_zero_with_neighborlist(
    kernels: List[Callable],
    cutoff: float,
    box_lengths: npt.ArrayLike,
    pbcs: npt.ArrayLike,
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, Callable]:
    """Get functions handling neighbor list, and computing direct energy part.

    Args:
        kernels: List of one-argument functions of a scalar argument,
            representing the interaction kernels at all levels.
        cutoff: Cutoff radius of the level-zero kernel (which is evaluated
            directly rather than by interpolation from grids).
        box_lengths: Sequence of side lengths of simulation box, one per
            spatial dimension.
        pbcs: Sequence of boolean values indicating whether the system is
            periodic along the corresponding direction.
        **neighbor_kwargs: Will be passed to `jax_md.partition.neighbor_list`.

    Returns:
        Tuple containing:
            - An instance of `jax_md.partition.NeighborListFns` that in turn
                consists of functions to allocate and update a neighbor list.
            - A function for calculating the direct energy contribution from
                given particle positions, charges, and a neighbor list.

    """
    box_lengths = jnp.asarray(box_lengths)
    pbcs = jnp.asarray(pbcs)

    # TODO: make mixed boundary conditions work
    if not (jnp.all(pbcs) or jnp.all(~pbcs)):
        raise ValueError("Mixed boundary conditions currently not supported.")
    periodic = pbcs[0]
    if periodic and cutoff > 0.5 * min(box_lengths):
        raise ValueError("Cutoff must not exceed half the shortest box length")

    k_0 = kernels[0]
    k_0_prime = jax.grad(k_0)
    sum_of_higher_kernels_at_zero = jnp.sum(
        jnp.asarray([k(0.0) for k in kernels[1:]])
    )

    if periodic:
        displacement_fn, shift_fn = space.periodic(box_lengths)
    else:
        displacement_fn, shift_fn = space.free()
    neighbor_fn = partition.neighbor_list(
        displacement_fn, box_lengths, r_cutoff=cutoff, **neighbor_kwargs
    )

    n_dim = box_lengths.shape[0]

    def compute_pair_distance_vectors(positions):
        # With the extra minus, the element `(i, j)` of the output is equal
        # to (under boundary conditions) `positions[i] - positions[j]`
        return -space.map_product(displacement_fn)(positions, positions)

    def pair_eval_fun(R_ij, q_i, q_j):
        r_ij = jnp.linalg.norm(R_ij)
        energy_contrib = q_i * q_j * k_0(jnp.linalg.norm(R_ij))
        force_contrib = -q_i * q_j * k_0_prime(r_ij) * R_ij / r_ij
        return jnp.concatenate([energy_contrib.reshape(1), force_contrib])

    eval_shortrange_all_pairs_neighborlist = make_eval_all_pairs_neighborlist(
        pair_distance_vector_fun=compute_pair_distance_vectors,
        pair_eval_fun=pair_eval_fun,
        pair_eval_fun_output_shape=pair_eval_fun(
            jnp.ones(n_dim), 1.0, 1.0
        ).shape,
    )

    def compute_U_and_f_zero_with_neighborlist(
        positions: jax.Array,
        charges: jax.Array,
        neighbor_indices: jax.Array,
    ) -> jax.Array:
        """Compute the direct energy contribution ($U^0$ in the article).

        Args:
            positions: Particle positions in cartesian coordinates,
                array of shape `(n_particles, n_dim)`.
            charges: Particle charges, array of shape `(n_particles,)`.
            neighbor_indices: Array of shape `(n_particles, n_max_neighbors)`,
                where the `i`-th row contains the indices of the neighbors of
                particle `i`. Index values greater than or equal to the total
                number of particles are interpreted as fill values (padding to
                consistent neighbor list shape plus spare capacity), and
                ignored in the energy evaluation.

        Returns:
            Tuple containing:
                - energy contribution
                - force contribution
        """
        pairwise_outputs = eval_shortrange_all_pairs_neighborlist(
            positions, charges, neighbor_indices
        )
        energy_pair_term = 0.5 * jnp.sum(pairwise_outputs[..., 0])
        energy_self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        energy = energy_pair_term - energy_self_interaction_term
        forces = jnp.sum(pairwise_outputs[..., 1:], axis=1)

        return energy, forces

    return neighbor_fn, compute_U_and_f_zero_with_neighborlist


def make_pair_term_fn(kernel_fn: Callable, pbc: npt.ArrayLike):
    pbc = onp.asarray(pbc)

    def apply_mic(deltas, cell):
        scaled = deltas @ jnp.linalg.pinv(cell)
        return jnp.where(pbc, (scaled - jnp.rint(scaled)) @ cell, deltas)

    def compute_pair_term(positions, charges, cell, neighbor_list):
        n_particles = positions.shape[0]
        is_not_placeholder = jnp.logical_and(
            neighbor_list[0] < n_particles, neighbor_list[1] < n_particles
        )
        dR = positions[neighbor_list[1]] - positions[neighbor_list[0]]
        dR = apply_mic(dR, cell)
        dr = jnp.linalg.norm(dR, axis=1)
        # Set distances of placeholder pairs to a value at which the potential
        # can be safely evaluated
        dr = jnp.where(is_not_placeholder, dr, 1.0)
        qi_qj = charges[neighbor_list[0]] * charges[neighbor_list[1]]
        return jnp.where(is_not_placeholder, qi_qj * kernel_fn(dr), 0.0).sum()

    return compute_pair_term


def make_compute_U0(kernel_fns: List[Callable], pbc: npt.ArrayLike):
    # TODO: Where to put the neighbor list option? Argument to this function,
    #  or make two separate functions?
    compute_pair_term = make_pair_term_fn(kernel_fn=kernel_fns[0], pbc=pbc)
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    def compute_U0(positions, charges, cell):
        # TODO: Replicate positions, charges, cell before passing to `compute_pair_term`
        n_centers = positions.shape[0]
        n_total = positions.shape[1]
        # TODO: Should the construction of these pair indices be put into a
        #  separate function (for isolated testing)?
        trivial_all_pairs_neighbor_list = onp.where(
            onp.arange(n_centers)[:, onp.newaxis] < onp.arange(n_total)
        )
        pair_term = compute_pair_term(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_list=trivial_all_pairs_neighbor_list,
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_U0


def make_compute_U0_neighbor_list(
    kernel_fns: List[Callable], pbc: npt.ArrayLike
):
    # TODO: Where to put the neighbor list option? Argument to this function,
    #  or make two separate functions?
    compute_pair_term = make_pair_term_fn(kernel_fn=kernel_fns[0], pbc=pbc)
    sum_of_higher_kernels_at_zero = onp.sum([k(0.0) for k in kernel_fns[1:]])

    def compute_U0_neighbor_list(positions, charges, cell, neighbor_list):
        # TODO: Replicate positions, charges, cell before passing to `compute_pair_term`
        pair_term = compute_pair_term(
            positions=positions,
            charges=charges,
            cell=cell,
            neighbor_list=neighbor_list,
        )
        self_interaction_term = (
            0.5 * jnp.sum(charges * charges) * sum_of_higher_kernels_at_zero
        )
        return pair_term - self_interaction_term

    return compute_U0_neighbor_list


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
