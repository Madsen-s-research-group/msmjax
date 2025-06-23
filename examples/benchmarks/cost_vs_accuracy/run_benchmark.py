import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from jaxlib.xla_extension import XlaRuntimeError
from matplotlib import pyplot as plt
from matscipy.neighbours import neighbour_list
from tqdm import tqdm

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import make_timed_eval, path_input_structures

# TODO: Command-line args or parameter file for all these things?

N_PARTICLES = 10000
PBC = (False, False, False)
# TODO: Explicitly compute as the average particle spacing instead?
LEVEL_ONE_SPACING = 1.0
QUANTITY = "energy"

LIST_OF_PS = [4, 6, 8]


def build_neighborlists(set_of_positions, set_of_cells, cutoff, pbc):
    neighbor_lists = []
    sizes = []
    n_structures = len(set_of_positions)
    print(f"- Building neighbor lists for {n_structures} structures")
    for pos, cll in tqdm(
        zip(set_of_positions, set_of_cells), total=n_structures
    ):
        nbl = neighbour_list(
            "ij", cutoff=cutoff, positions=pos, cell=cll, pbc=pbc
        )
        nbl_no_duplicates = jnp.unique(
            jnp.sort(jnp.column_stack([nbl[0], nbl[1]]), axis=1), axis=0
        )
        nbl_no_duplicates = (nbl_no_duplicates[:, 0], nbl_no_duplicates[:, 1])
        neighbor_lists.append(nbl_no_duplicates)
        sizes.append(len(nbl_no_duplicates[0]))
    target_size = max(sizes)
    n_particles = set_of_positions.shape[1]
    placeholder_index = n_particles
    for idx_structure in range(len(neighbor_lists)):
        i, j = neighbor_lists[idx_structure]
        padding = target_size - len(i)
        neighbor_lists[idx_structure] = (
            jnp.pad(i, (0, padding), constant_values=placeholder_index),
            jnp.pad(j, (0, padding), constant_values=placeholder_index),
        )
    print("- Finished building neighbor lists")
    return neighbor_lists


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
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

    baseoutdir = Path(cmd_args.outdir)
    baseoutdir.mkdir(parents=True)
    npz_file = path_input_structures / f"structures_{N_PARTICLES}.npz"
    structures = onp.load(npz_file)
    n_structures = len(structures["cells"])

    resultsfile = baseoutdir / "results.csv"

    # TODO: This restriction is only sensible and necessary in non-periodic case
    cell = jax.device_put(structures["cells"][0])
    half_sidelength = 0.5 * cell[0, 0]
    range_of_alphas = onp.arange(
        3.0, min(half_sidelength / LEVEL_ONE_SPACING, 8.01), 1.0
    )
    range_of_cutoffs = range_of_alphas * LEVEL_ONE_SPACING

    print(f"- Range of cutoffs is {range_of_cutoffs}")
    print()

    for level_zero_cutoff in range_of_cutoffs:
        print("#" * 80)
        print(f"- r_cut_0 = {level_zero_cutoff:.2f}")
        print("#" * 80)
        neighborlists = build_neighborlists(
            structures["positions"],
            structures["cells"],
            level_zero_cutoff,
            pbc=PBC,
        )
        print()
        for p in LIST_OF_PS:
            print(f"- p = {p}:")
            msm_params = set_up_msm_params(
                cell=cell,
                level_one_spacings=LEVEL_ONE_SPACING,
                level_zero_cutoff=level_zero_cutoff,
                p=p,
                pbc=PBC,
                cell_mode="ortho",
                dynamic_cell=False,
                n_particles=N_PARTICLES,
                use_neighborlist=True,
                neighborlist_prefactor=0.5,  # for no-duplicate neighbor list
            )
            msm_evaluation_fns = create_msm(msm_params)
            timing_fn = make_timed_eval(
                msm_evaluation_fns[QUANTITY], repeat=5, number=50
            )
            all_times = []
            all_calculation_results = []
            print(f"- Starting timing loop over {n_structures} structures")
            for idx_structure in tqdm(range(n_structures)):
                # TODO: double precision?
                pos = jax.device_put(structures["positions"][idx_structure])
                chg = jax.device_put(structures["charges"][idx_structure])
                nbl = jax.device_put(neighborlists[idx_structure])
                # TODO: try-except out-of-memory errors?
                min_time, calculation_result = timing_fn(
                    pos, chg, neighborlist=nbl
                )
                all_times.append(min_time)
                all_calculation_results.append(calculation_result)

            # TODO: What (else) to save? quantity? pbc?
            results_tmp = pd.DataFrame(
                data={
                    "n_particles": N_PARTICLES,
                    "level_zero_cutoff": level_zero_cutoff,
                    "p": p,
                    "time": onp.mean(all_times),
                    # "error": error, # TODO
                },
                index=[0],
            )
            print(f"- Writing results to {resultsfile}.")
            if not resultsfile.is_file():
                results_tmp.to_csv(resultsfile, index=False, mode="w")
            else:
                results_tmp.to_csv(
                    resultsfile, index=False, mode="a", header=False
                )
