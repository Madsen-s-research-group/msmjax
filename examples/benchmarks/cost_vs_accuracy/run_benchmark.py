import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matscipy.neighbours
import numpy as onp
import pandas as pd
from jaxlib.xla_extension import XlaRuntimeError
from tqdm import tqdm

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import (
    calc_relative_rmse,
    make_timed_eval,
)

# TODO: Explicitly compute as the average particle spacing instead?
LEVEL_ONE_SPACING = 1.0

# TODO: add (option for) periodic and slab structures
# PBC = (False, False, False)
# INDIR = Path("reference_data/") / "nonperiodic"
# PBC = (True, True, True)
# INDIR = Path("reference_data/") / "periodic"

DATADIR = Path("reference_data")

LIST_OF_PS = [4, 6, 8]

LABELMAP_QUANTITIES = {
    "energy": "energies",
    "forces": "forces",
    "charge_gradient": "chargegrads",
    "stress": "stresses",
}
MAP_STRUCTURETYPES = {
    "nonperiodic": {"indir": DATADIR / "nonperiodic", "pbc": (False,) * 3},
    "periodic": {"indir": DATADIR / "periodic", "pbc": (True,) * 3},
}


@partial(jax.jit, static_argnums=2)
def remove_duplicates_from_neighborlist(neighborlist, fill_value, size):
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
    n_structures = len(set_of_positions)
    n_particles = set_of_positions.shape[1]

    neighborlists_raw = []
    print(f"- Building neighbor lists for {n_structures} structures")
    for pos, cll in tqdm(
        zip(set_of_positions, set_of_cells), total=n_structures
    ):
        nbl = matscipy.neighbours.neighbour_list(
            "ij", cutoff=cutoff, positions=pos, cell=cll, pbc=pbc
        )
        neighborlists_raw.append(nbl)

    # Pad to common max length
    max_size = max([len(nbl[0]) for nbl in neighborlists_raw])
    placeholder_index = n_particles
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
    print("- Done building neighbor lists")

    return neighborlists_nodupes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--structuretype",
        required=True,
        type=str,
        choices=["nonperiodic", "periodic"],
    )
    parser.add_argument(
        "--quantity",
        required=True,
        type=str,
        choices=LABELMAP_QUANTITIES.keys(),
        nargs="+",
        help="Quantity or quantities to evaluate for the benchmark",
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
    structuretype = cmd_args.structuretype
    structures = onp.load(
        MAP_STRUCTURETYPES[structuretype]["indir"] / "structures.npz"
    )
    n_structures = structures["positions"].shape[0]
    n_particles = structures["positions"].shape[1]
    reference_results = onp.load(
        MAP_STRUCTURETYPES[structuretype]["indir"] / "reference_results.npz"
    )
    pbc = MAP_STRUCTURETYPES[structuretype]["pbc"]

    outfile = baseoutdir / "results.csv"

    # All structures are assumed to have the same cell
    cell = jax.device_put(structures["cells"][0])
    side_lengths = onp.linalg.norm(cell, axis=1)
    range_of_alphas = onp.arange(3.0, 8.01, 1.0)
    # TODO: If I add slab structures, add a similar check that only takes the
    #  periodic x-y directions into account
    if onp.array(pbc).all() or (onp.logical_not(pbc)).all():
        range_of_alphas = range_of_alphas[
            range_of_alphas <= 0.5 * max(side_lengths) / LEVEL_ONE_SPACING
        ]
    range_of_cutoffs = range_of_alphas * LEVEL_ONE_SPACING

    print(f"- Range of cutoffs is {range_of_cutoffs}")
    print()

    for level_zero_cutoff in range_of_cutoffs:
        print("#" * 80)
        print(f"- r_cut_0 = {level_zero_cutoff:.2f}")
        print("#" * 80)
        neighborlists = build_duplicate_free_neighborlists(
            structures["positions"],
            structures["cells"],
            level_zero_cutoff,
            pbc=pbc,
        )
        print()
        for p in LIST_OF_PS:
            print(f"- p = {p}:")
            msm_params = set_up_msm_params(
                cell=cell,
                level_one_spacings=LEVEL_ONE_SPACING,
                level_zero_cutoff=level_zero_cutoff,
                p=p,
                pbc=pbc,
                cell_mode="ortho",
                dynamic_cell=False,
                n_particles=n_particles,
                use_neighborlist=True,
                neighborlist_prefactor=1.0,  # for no-duplicate neighbor list
            )
            msm_evaluation_fns = create_msm(msm_params)
            # TODO: Add "repeat" and "number" as command-line args?
            for quantity in cmd_args.quantity:
                print(f"- Evaluating quantity: {quantity}")
                timing_fn = make_timed_eval(
                    msm_evaluation_fns[quantity], repeat=5, number=10
                )
                all_times = []
                all_calculation_results = []
                print(f"- Starting timing loop over {n_structures} structures")
                for idx_structure in tqdm(range(n_structures)):
                    pos = jax.device_put(
                        structures["positions"][idx_structure]
                    )
                    chg = jax.device_put(structures["charges"][idx_structure])
                    nbl = jax.device_put(neighborlists[idx_structure])
                    # TODO: try-except out-of-memory errors?
                    min_time, calculation_result = timing_fn(
                        pos, chg, neighborlist=nbl
                    )
                    # TODO: if quantity == "stress", reduce to 6-component format
                    all_times.append(min_time)
                    all_calculation_results.append(calculation_result)
                all_times = jnp.array(all_times)
                all_calculation_results = jnp.array(all_calculation_results)

                relative_rmse = calc_relative_rmse(
                    all_calculation_results,
                    reference_results[LABELMAP_QUANTITIES[quantity]],
                )
                results_tmp = pd.DataFrame(
                    data={
                        "n_particles": n_particles,
                        "level_zero_cutoff": level_zero_cutoff,
                        "p": p,
                        "quantity": quantity,
                        "time": all_times.mean(),
                        "relative_rmse": relative_rmse,
                    },
                    index=[0],
                )
                print(f"- Writing results to {outfile}.")
                if not outfile.is_file():
                    results_tmp.to_csv(outfile, index=False, mode="w")
                else:
                    results_tmp.to_csv(
                        outfile, index=False, mode="a", header=False
                    )
            print()
        print()

    print(
        "- Finished running benchmark. "
        "You can use 'make_plots.ipynb' to plot the results."
    )
