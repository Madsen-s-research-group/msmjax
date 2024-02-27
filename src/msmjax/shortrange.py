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

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
import numpy.typing as npt
from jax_md import partition, space

from msmjax.kernels import SofteningFunctionOneOverR, split_one_over_r_kernel


def make_evaluate_shortrange_with_neighbor_list(
    shortrange_kernel: Callable,
    cutoff: float,
    box: npt.ArrayLike,
    pbcs: npt.ArrayLike,
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, Callable]:
    box = jnp.asarray(box)
    pbcs = jnp.asarray(pbcs)

    # TODO: if periodic, check if cutoff <= half of box size

    if not (jnp.all(pbcs) or jnp.all(~pbcs)):
        raise ValueError("Mixed boundary conditions currently not supported.")
    periodic = pbcs[0]

    if periodic:
        displacement_fn, shift_fn = space.periodic(box)
    else:
        displacement_fn, shift_fn = space.free()

    neighbor_fn = partition.neighbor_list(
        displacement_fn, box, r_cutoff=cutoff, **neighbor_kwargs
    )

    def evaluate_one_row_of_neighborlist(idx_of_row, row, dR):
        nb_particles = dR.shape[0]
        pair_contribs = jnp.where(
            row < nb_particles,
            shortrange_kernel(jnp.linalg.norm(dR[idx_of_row, row], axis=1)),
            0.0,
        )

        return jnp.sum(pair_contribs)

    def energy_fn(positions, neighborlist):
        dR = space.map_product(displacement_fn)(positions, positions)

        all_indices = jnp.arange(neighborlist.idx.shape[0])

        energy = 0.5 * jnp.sum(
            jax.vmap(evaluate_one_row_of_neighborlist, (0, 0, None), 0)(
                all_indices, neighborlist.idx, dR
            )
        )

        return energy

    return neighbor_fn, energy_fn


if __name__ == "__main__":
    sidelength = 10.0
    ndim = 3
    level_zero_cutoff = 3.75
    max_gridlevel = 4
    p = 4
    n_particles = 15
    pbcs = [True] * ndim
    box = jnp.array([sidelength] * 3)

    rng = onp.random.default_rng(58347)
    pos = rng.uniform(low=0.0, high=sidelength, size=(n_particles, ndim))
    chg = rng.uniform(low=-1.0, high=1.0, size=n_particles)

    print(pos)
    print(chg)

    kernels = split_one_over_r_kernel(
        max_level=max_gridlevel,
        level_zero_cutoff=level_zero_cutoff,
        softening_function=SofteningFunctionOneOverR(p),
    )

    neighbor_fn, energy_fn = make_evaluate_shortrange_with_neighbor_list(
        shortrange_kernel=kernels[0],
        cutoff=level_zero_cutoff,
        box=box,
        pbcs=pbcs,
    )
    nbl_allocate_fun = neighbor_fn.allocate
    nbl_update_fun = neighbor_fn.update
    jitted_nbl_update_fun = jax.jit(nbl_update_fun)
    jitted_eval_direct_energy = jax.jit(energy_fn)

    neighborlist = nbl_allocate_fun(pos)

    e = jitted_eval_direct_energy(pos, neighborlist)
    print(e)

    def calculate_energy_reference_loop(positions, charges):
        out = 0.0
        for i in range(positions.shape[0]):
            for j in range(i):
                r_ij = onp.linalg.norm(positions[i] - positions[j])
                out += charges[i] * charges[j] * kernels[0](r_ij)

        return out

    e_ref = calculate_energy_reference_loop(pos, chg)
    print(e_ref)
