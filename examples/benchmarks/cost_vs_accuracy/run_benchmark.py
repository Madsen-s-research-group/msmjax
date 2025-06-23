import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from pathlib import Path

import jax
import numpy as onp
import pandas as pd
from jaxlib.xla_extension import XlaRuntimeError
from matplotlib import pyplot as plt
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

    # TODO: This restriction is only meaningful and necessary in non-periodic case
    cell = jax.device_put(structures["cells"][0])
    half_sidelength = 0.5 * cell[0, 0]
    range_of_alphas = onp.arange(3.0, half_sidelength / LEVEL_ONE_SPACING, 1.0)
    range_of_cutoffs = range_of_alphas * LEVEL_ONE_SPACING

    for p in LIST_OF_PS:
        print(f"- Looping over cutoffs for {p=}")
        for level_zero_cutoff in tqdm(range_of_cutoffs):
            msm_params = set_up_msm_params(
                cell=cell,
                level_one_spacings=LEVEL_ONE_SPACING,
                level_zero_cutoff=level_zero_cutoff,
                p=p,
                pbc=PBC,
                cell_mode="ortho",
                dynamic_cell=False,
                n_particles=N_PARTICLES,
            )
            msm_evaluation_fns = create_msm(msm_params)
            timing_fn = make_timed_eval(
                msm_evaluation_fns[QUANTITY], repeat=5, number=50
            )
            all_times = []
            all_calculation_results = []
            for idx_structure in range(n_structures):
                pos = jax.device_put(structures["positions"][idx_structure])
                chg = jax.device_put(structures["charges"][idx_structure])
                # TODO: neighbor list
                min_time, calculation_result = timing_fn(pos, chg)
                all_times.append(min_time)
                all_calculation_results.append(calculation_result)

            print(all_times)
            # TODO: save output
