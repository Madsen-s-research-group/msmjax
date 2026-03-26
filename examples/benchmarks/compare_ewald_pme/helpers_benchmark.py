# Copyright 2026 The msmJAX contributors
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: Contains code from jax_md._energy.electrostatics.py, v0.2.25, modified
# by the msmJAX contributors in order to fix suspected minor bugs.

"""Helpers for benchmarking msmJAX vs. JAX-MD electrostatics v0.2.25"""

from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax_md import partition, space, util
from jax_md._energy.electrostatics import (
    coulomb_direct_neighbor_list,
    coulomb_recip_pme,
    structure_factor,
)

from msmjax.core.shortrange import _gen_supercell
from msmjax.utils.benchmarking import path_input_structures

DATADIR = Path("prepared_data/")
REF_FORCE_DIR = DATADIR / "reference_forces/"


REPETITIONS = onp.array([1, 2, 3, 4, 5])
NUMS_PARTICLES_ORIGINAL = [
    onp.array(
        [
            500,
            1000,
            1500,
            2000,
            2500,
            3000,
            3500,
            4500,
            5000,
            6000,
            8000,
            10000,
            12000,
            15000,
        ]
    ),
    onp.array([2500, 3500, 4500, 6000, 8000, 10000]),
    onp.array([4000, 5000, 7000, 9000, 12000, 15000]),
    onp.array([8000, 10000, 12000, 15000]),
    onp.array([10000, 12000]),
]

all_repetitions = [
    onp.full_like(nums, r)
    for r, nums in zip(REPETITIONS, NUMS_PARTICLES_ORIGINAL)
]
all_nums_particles_final = [
    reps**3 * nums for reps, nums in zip(REPETITIONS, NUMS_PARTICLES_ORIGINAL)
]
all_repetitions = onp.concatenate(all_repetitions)
all_nums_particles_final = onp.concatenate(all_nums_particles_final)
all_num_particles_original = onp.concatenate(NUMS_PARTICLES_ORIGINAL)


def iter_structures_and_forces(
    request_n_particles: Sequence[int] = None, max_n_particles: int = None
):
    match (request_n_particles, max_n_particles):
        case (None, None):
            inds = onp.arange(len(all_repetitions))
        case (None, _):
            inds = onp.arange(len(all_repetitions))[
                all_nums_particles_final <= max_n_particles
            ]
        case (_, None):
            if not onp.isin(
                request_n_particles, all_nums_particles_final
            ).all():
                raise ValueError(
                    "Requested a number of particles "
                    "for which no structure is available."
                )
            inds = onp.where(
                (
                    all_nums_particles_final[:, onp.newaxis]
                    == request_n_particles
                ).any(axis=1)
            )[0]
        case _:
            raise ValueError("Mutually exclusive arguments.")

    for rep, n_particles_original in zip(
        all_repetitions[inds], all_num_particles_original[inds]
    ):
        structures = onp.load(
            path_input_structures / f"structures_{n_particles_original}.npz"
        )
        pos = structures["positions"][0]
        chg = structures["charges"][0]
        cll = structures["cells"][0]
        force_data = onp.load(
            REF_FORCE_DIR / f"forces_singleprec_{n_particles_original}.npz"
        )
        forces = force_data["forces"]
        f_std = force_data["std"]
        if rep > 1:
            pos, chg, cll = _gen_supercell(
                pos, chg, cll, supercell_diag=[rep] * 3
            )
            pos, chg, cll = onp.array(pos), onp.array(chg), onp.array(cll)
            forces = onp.tile(forces, (rep**3, 1))
        yield pos, chg, cll, forces, f_std


def compute_ewald_self_energy(charges, alpha):
    """In order to be able to compare Ewald/PME energies with MSM.
    Does not matter for forces.
    """
    return alpha / jnp.sqrt(jnp.pi) * (charges * charges).sum()


@partial(jax.jit, static_argnums=(1,))
def custom_safe_mask(
    mask, fn, operand, placeholder_in=1.0, placeholder_out=0.0
):
    masked = jnp.where(mask, operand, placeholder_in)
    return jnp.where(mask, fn(masked), placeholder_out)


def custom_coulomb_recip_ewald(
    charge: Array, side_length: Array, alpha: float, g_max: float
) -> Callable[[Array], Array]:
    """Adapted version of jax_md._energy.electrostatics.coulomb_recip_ewald
    that fixes two suspected minor bugs.
    """

    def energy_fn(position, **kwargs):
        dim = position.shape[-1]
        V = side_length**dim

        dg = 2 * onp.pi / side_length
        # Just to make the sum inclusive.
        g_range = onp.arange(0, g_max + dg / 2, dg)
        g_range = onp.concatenate((-g_range[::-1], g_range[1:]))

        gx, gy, gz = jnp.meshgrid(g_range, g_range, g_range)
        g = jnp.reshape(jnp.stack((gx, gy, gz), axis=-1), (-1, dim))
        g2 = jnp.sum(g**2, axis=-1)
        mask = (g2 < g_max**2) & (g2 > 1e-7)

        # FB: changed by a factor of two (was (4 * jnp.pi) / V originally)
        Z = (2 * jnp.pi) / V
        S2 = jnp.abs(structure_factor(g, position, charge)) ** 2
        fn = lambda g2: jnp.exp(-g2 / (4 * alpha**2)) / g2 * S2

        # FB: replaced the masking functions to avoid nan derivatives
        return Z * util.high_precision_sum(custom_safe_mask(mask, fn, g2))

    return energy_fn


def set_up_ewald_jaxmd_dynalpha(
    side_length: float, cutoff: float, g_max: float, **neighbor_kwargs
):
    """Convenience wrapper for setting up Ewald evaluation functions.

    Similar to jax_md._energy.electrostatics.coulomb_ewald_neighbor_list,
    but using adapted reciprocal-space part that fixes suspected minor bugs
    in original version, and promoting alpha to a dynamic argument.
    """
    displacement_fn, _ = space.periodic(side_length)

    neighbor_fn = partition.neighbor_list(
        space.canonicalize_displacement_or_metric(displacement_fn),
        side_length,
        cutoff,
        **neighbor_kwargs,
    )

    def calc_energy(positions, charges, neighborlist, alpha):
        _, direct_fn = coulomb_direct_neighbor_list(
            displacement_fn,
            side_length,
            charges,
            alpha=alpha,
            cutoff=cutoff,
        )
        recip_fn = custom_coulomb_recip_ewald(
            charges, side_length, alpha=alpha, g_max=g_max
        )
        return (
            direct_fn(positions, neighborlist)
            + recip_fn(positions)
            - compute_ewald_self_energy(charges, alpha)
        )

    def calc_forces(positions, charges, neighborlist, alpha):
        return -jax.grad(calc_energy, argnums=0)(
            positions, charges, neighborlist, alpha
        )

    evaluation_functions = {"energy": calc_energy, "forces": calc_forces}

    return neighbor_fn, evaluation_functions


def set_up_pme_jaxmd_dynalpha(
    side_length: float, cutoff: float, grid_points: int, **neighbor_kwargs
):
    """Convenience wrapper for setting up PME evaluation functions.

    Similar to jax_md._energy.electrostatics.coulomb_neighbor_list,
    but promoting alpha to a dynamic argument.
    """
    displacement_fn, _ = space.periodic(side_length)

    neighbor_fn = partition.neighbor_list(
        space.canonicalize_displacement_or_metric(displacement_fn),
        side_length,
        cutoff,
        **neighbor_kwargs,
    )

    def calc_energy(positions, charges, neighborlist, alpha):
        _, direct_fn = coulomb_direct_neighbor_list(
            displacement_fn,
            side_length,
            charges,
            alpha=alpha,
            cutoff=cutoff,
        )
        recip_fn = coulomb_recip_pme(
            charges, side_length, alpha=alpha, grid_points=grid_points
        )
        return (
            direct_fn(positions, neighborlist)
            + recip_fn(positions)
            - compute_ewald_self_energy(charges, alpha)
        )

    def calc_forces(positions, charges, neighborlist, alpha):
        return -jax.grad(calc_energy, argnums=0)(
            positions, charges, neighborlist, alpha
        )

    evaluation_functions = {"energy": calc_energy, "forces": calc_forces}

    return neighbor_fn, evaluation_functions
