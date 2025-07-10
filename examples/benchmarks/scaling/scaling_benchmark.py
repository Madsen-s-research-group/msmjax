"""Demonstration of scaling of the MSM implementation with particle number."""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matscipy.neighbours
import numpy as onp
import pandas as pd
from tqdm import tqdm

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.core.shortrange import _gen_supercell, make_eval_pair_pot
from msmjax.utils.benchmarking import (
    calc_relative_rmse,
    make_timed_eval,
    path_input_structures,
)

# TODO: Explicitly compute as the average particle spacing instead?
LEVEL_ONE_SPACING = 1.0
# TODO: Also run, or offer the option to run, with periodicity
PBC = (False, False, False)
# TODO: define where? value(s)?
LEVEL_ZERO_CUTOFF = 3.0
P = 4
# TODO: energy, other quantities?
QUANTITY = "forces"


def coulomb_kernel(r):
    return 1.0 / r


def calc_nonperiodic_ref_energy(positions, charges):
    # TODO: name
    n_dim = positions.shape[1]
    compute_pair_term = make_eval_pair_pot(
        kernel_fn=coulomb_kernel, pbc=(False,) * n_dim
    )
    return compute_pair_term(positions, charges)


def calc_nonperiodic_ref_forces(positions, charges):
    # TODO: name
    return -jax.grad(calc_nonperiodic_ref_energy, argnums=0)(
        positions, charges
    )


@partial(jax.jit, static_argnums=2)
def remove_duplicates_from_neighborlist(neighborlist, fill_value, size):
    # TODO: move to utils (used both here and in cost_vs_accuracy benchmark)
    without_duplicates = jnp.unique(
        jnp.sort(jnp.column_stack([neighborlist[0], neighborlist[1]]), axis=1),
        axis=0,
        size=size,
        fill_value=fill_value,
    )
    return (without_duplicates[:, 0], without_duplicates[:, 1])


def build_duplicate_free_neighborlists(
    set_of_positions, set_of_cells, cutoff, pbc
):
    # TODO: move to utils (used both here and in cost_vs_accuracy benchmark)
    n_structures = len(set_of_positions)

    neighborlists_raw = []
    print(f"- Building neighbor list(s) for {n_structures} structure(s)")
    for pos, cll in tqdm(
        zip(set_of_positions, set_of_cells), total=n_structures
    ):
        nbl = matscipy.neighbours.neighbour_list(
            "ij", cutoff=cutoff, positions=pos, cell=cll, pbc=pbc
        )
        neighborlists_raw.append(nbl)

    # Pad to common max length
    max_size = max([len(nbl[0]) for nbl in neighborlists_raw])
    placeholder_index = max(pos.shape[0] for pos in set_of_positions)
    for idx_structure in range(n_structures):
        i, j = neighborlists_raw[idx_structure]
        padding = max_size - len(i)
        neighborlists_raw[idx_structure] = (
            jnp.pad(i, (0, padding), constant_values=placeholder_index),
            jnp.pad(j, (0, padding), constant_values=placeholder_index),
        )

    max_size_nodupes = max_size // 2
    neighborlists_nodupes = [
        remove_duplicates_from_neighborlist(
            nbl, fill_value=placeholder_index, size=max_size_nodupes
        )
        for nbl in neighborlists_raw
    ]
    print("- Done building neighbor list(s)")

    return neighborlists_nodupes


def structure_generator():
    # TODO: name
    for repeats, unrepeated_particle_nums in zip(
        [None, 2, 3],
        [
            [3000, 6000, 9000, 12000, 15000],
            [2500, 3500, 4500, 6000, 8000, 10000],
            [4000, 5000, 7000, 9000, 12000, 15000],
        ],
    ):
        for n_particles_original in unrepeated_particle_nums:
            structures = onp.load(
                path_input_structures
                / f"structures_{n_particles_original}.npz"
            )
            # TODO: .astype(onp.float64)?
            # TODO: jax.device_put (where?)
            pos = jax.device_put(structures["positions"][0])
            chg = jax.device_put(structures["charges"][0])
            cell = jax.device_put(structures["cells"][0])
            if repeats is not None:
                pos, chg, cell = _gen_supercell(
                    pos, chg, cell, supercell_diag=[repeats] * 3
                )
            yield pos, chg, cell


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Demonstration of scaling of the MSM implementation "
        "with particle number."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Output directory. Existing outputs will not be overwritten.",
    )
    parser.add_argument(
        "--jax_enable_x64",
        action="store_true",
        default=False,
        help="Flag indicating that double precision should be used",
    )
    cmd_args = parser.parse_args()

    if cmd_args.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("- Running JAX in double-precision mode.")

    baseoutdir = Path(cmd_args.outdir)
    baseoutdir.mkdir(parents=True)
    outfile = baseoutdir / "results.csv"

    for pos, chg, cell in structure_generator():
        n_particles = pos.shape[0]
        print(f"- n_particles = {n_particles}")
        neighborlist = build_duplicate_free_neighborlists(
            [pos], [cell], LEVEL_ZERO_CUTOFF, pbc=PBC
        )[0]
        msm_params = set_up_msm_params(
            cell=cell,
            level_one_spacings=LEVEL_ONE_SPACING,
            level_zero_cutoff=LEVEL_ZERO_CUTOFF,
            p=P,
            pbc=PBC,
            cell_mode="ortho",
            dynamic_cell=False,
            n_particles=n_particles,
            use_neighborlist=True,
            neighborlist_prefactor=1.0,  # duplicate-free neighbor list
        )
        msm_evaluation_fns = create_msm(msm_params)
        # TODO: Add "repeat" and "number" as command-line args?
        timing_fn = make_timed_eval(
            msm_evaluation_fns[QUANTITY], repeat=10, number=15
        )
        min_time, _ = timing_fn(pos, chg, neighborlist=neighborlist)
        print(f"- time = {min_time * 1000:.2f} ms")

        results_tmp = pd.DataFrame(
            data={
                "n_particles": n_particles,
                "level_zero_cutoff": LEVEL_ZERO_CUTOFF,
                "p": P,
                "quantity": QUANTITY,
                "time": min_time,
            },
            index=[0],
        )
        print(f"- Writing results to {outfile}.")
        if not outfile.is_file():
            results_tmp.to_csv(outfile, index=False, mode="w")
        else:
            results_tmp.to_csv(outfile, index=False, mode="a", header=False)

        print()
