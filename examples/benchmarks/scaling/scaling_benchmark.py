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

try:
    # TODO: This might not actually solve the problem in newer JAX versions.
    #  Check which exception is actually raised in newer versions when running
    #  out of memory, it might not be the same one!
    from jaxlib._jax import XlaRuntimeError
except ModuleNotFoundError:
    # To work with older JAX versions
    from jaxlib.xla_extension import XlaRuntimeError

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.core.shortrange import _gen_supercell, make_eval_pair_pot
from msmjax.utils.benchmarking import (
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


exact_nonperiodic_evaluation_fns = {
    "energy": calc_nonperiodic_ref_energy,
    "forces": calc_nonperiodic_ref_forces,
}


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
        [None, 2, 3, 4],
        [
            [3000, 6000, 9000, 12000, 15000],
            [2500, 3500, 4500, 6000, 8000, 10000],
            [4000, 5000, 7000, 9000, 12000, 15000],
            [8000, 10000, 12000],
        ],
    ):
        for n_particles_original in unrepeated_particle_nums:
            structures = onp.load(
                path_input_structures
                / f"structures_{n_particles_original}.npz"
            )
            # TODO: .astype(onp.float64)? (will prob need to be an argument to
            #  the generator)
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
    # TODO: name of this argument
    # TODO: raise an error if used together with periodic boundary conditions
    parser.add_argument(
        "--exact",
        action="store_true",
        default=False,
        help="Flag indicating not to use MSM, but exact all-pairs evaluation, "
        "for comparison. Only available in combination with non-periodic "
        "boundary conditions.",
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
    resultsfile = baseoutdir / "results.csv"

    for pos, chg, cell in structure_generator():
        n_particles = pos.shape[0]
        print(f"- n_particles = {n_particles}")

        if cmd_args.exact:
            fn = exact_nonperiodic_evaluation_fns[QUANTITY]
        else:
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
            fn = msm_evaluation_fns[QUANTITY]

        try:
            # TODO: Add "repeat" and "number" as command-line args?
            timing_fn = make_timed_eval(fn, repeat=10, number=15)
            if cmd_args.exact:
                time, _ = timing_fn(pos, chg)
            else:
                neighborlist = build_duplicate_free_neighborlists(
                    [pos], [cell], LEVEL_ZERO_CUTOFF, pbc=PBC
                )[0]
                time, _ = timing_fn(pos, chg, neighborlist=neighborlist)
        except XlaRuntimeError as e:
            if "out of memory" in str(e).lower():
                print("- Out of memory: skipping the rest of the loop.")
                break
            else:
                raise e

        print(f"- time = {time * 1000:.2f} ms")

        outdata = {
            "n_particles": n_particles,
            "quantity": QUANTITY,
            "time": time,
        }
        if cmd_args.exact:
            outdata["exact"] = True  # TODO: name?
        else:
            outdata["exact"] = False  # TODO: name?
            outdata["level_zero_cutoff"] = LEVEL_ZERO_CUTOFF
            outdata["p"] = P
        results_tmp = pd.DataFrame(data=outdata, index=[0])
        print(f"- Writing results to {resultsfile}.")
        if not resultsfile.is_file():
            results_tmp.to_csv(resultsfile, index=False, mode="w")
        else:
            results_tmp.to_csv(
                resultsfile, index=False, mode="a", header=False
            )

        print()

    print()

    results = pd.read_csv(resultsfile)
    particle_numbers = results["n_particles"].values
    times = results["time"].values
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Time / ms")
    ax.scatter(particle_numbers, times * 1000)
    # TODO: legend?
    ax.set_xscale("log")
    ax.set_yscale("log")
    filename_without_suffix = f"scaling_{QUANTITY}_loglog"
    for suffix in [".png", ".pdf"]:
        outfile_plot = baseoutdir / (filename_without_suffix + suffix)
        print(f"- Saving plot to: {outfile_plot}")
        fig.savefig(outfile_plot)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    filename_without_suffix = f"scaling_{QUANTITY}"
    for suffix in [".png", ".pdf"]:
        outfile_plot = baseoutdir / (filename_without_suffix + suffix)
        print(f"- Saving plot to: {outfile_plot}")
        fig.savefig(outfile_plot)
