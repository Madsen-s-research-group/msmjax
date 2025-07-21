"""Analyze cost-accuracy tradeoff for different MSM parameter settings."""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from tqdm import tqdm

try:
    from jaxlib._jax import XlaRuntimeError
except ModuleNotFoundError:
    # To work with older JAX versions
    from jaxlib.xla_extension import XlaRuntimeError

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import (
    build_duplicate_free_neighborlists,
    calc_relative_rmse,
    inds_matrix_to_six_component_stress,
    make_timed_eval,
)

# TODO: Explicitly compute as the average particle spacing instead?
LEVEL_ONE_SPACING = 1.0

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
    "slab": {"indir": DATADIR / "slab", "pbc": (True, True, False)},
    "slab-nozdipole": {
        "indir": DATADIR / "slab-nozdipole",
        "pbc": (True, True, False),
    },
}


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Analyze cost-accuracy tradeoff for "
        "different MSM parameter settings."
    )
    parser.add_argument(
        "--structuretype",
        required=True,
        type=str,
        choices=MAP_STRUCTURETYPES.keys(),
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
    range_of_alphas = onp.arange(3, 13).astype(float)
    if not onp.array(pbc).any():
        # If non-periodic, limit the level-zero cutoffs to below half the side
        # length, as a larger one would incur an unnecessary amount of
        # short-range evaluations.
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
        try:
            neighborlists = build_duplicate_free_neighborlists(
                structures["positions"],
                structures["cells"],
                level_zero_cutoff,
                pbc=pbc,
            )
        except (XlaRuntimeError, ValueError) as e:
            if "out of memory" in str(e).lower():
                print("- Out of memory: skipping the remaining cutoff values.")
                print()
                break
            else:
                raise e
        print()
        for p in LIST_OF_PS:
            print(f"- p = {p}:")
            for quantity in cmd_args.quantity:
                print(f"- Evaluating quantity: {quantity}")
                use_dynamic_cell = quantity == "stress"
                if onp.any(pbc):
                    use_neighborlist = False
                    supercell_diag = onp.ceil(
                        2 * level_zero_cutoff / onp.diag(cell)
                    ).astype(int)
                    # TODO: double-check this works correctly for slabs
                    supercell_diag = onp.where(pbc, supercell_diag, 1)
                    neighborlist_prefactor = None
                else:
                    use_neighborlist = True
                    # A prefactor 1.0 corresponds to a duplicate-free
                    # neighbor list:
                    neighborlist_prefactor = 1.0
                    supercell_diag = None
                msm_params = set_up_msm_params(
                    cell=cell,
                    level_one_spacings=LEVEL_ONE_SPACING,
                    level_zero_cutoff=level_zero_cutoff,
                    p=p,
                    pbc=pbc,
                    cell_mode="ortho",
                    dynamic_cell=use_dynamic_cell,
                    n_particles=n_particles,
                    supercell_diag=supercell_diag,
                    use_neighborlist=use_neighborlist,
                    neighborlist_prefactor=neighborlist_prefactor,
                )
                msm_evaluation_fns = create_msm(msm_params)
                # TODO: Add "repeat" and "number" as command-line args?
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
                    if use_dynamic_cell:
                        min_time, calculation_result = timing_fn(
                            pos, chg, cell=cell, neighborlist=nbl
                        )
                    else:
                        min_time, calculation_result = timing_fn(
                            pos, chg, neighborlist=nbl
                        )
                    if quantity == "stress":
                        calculation_result = calculation_result[
                            inds_matrix_to_six_component_stress
                        ]
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

    print("- Finished running benchmark.")
    print()

    markerlist = ["x", "o", "s", "d", "v", "^"]
    # Settings for annotating the data points with the cutoff value:
    # See https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html#using-offset-transforms-to-create-a-shadow-effect for the handling of the annotation offsets from the data points
    ha = "center"
    va = "top"
    dx, dy = -3 / 72.0, -4 / 72.0  # x and y offsets in points

    results_all = pd.read_csv(outfile)
    for quantity in results_all["quantity"].unique():
        unique_ps = results_all.loc[
            results_all["quantity"] == quantity, "p"
        ].unique()
        unique_ps = onp.sort(unique_ps)

        fig, ax = plt.subplots()
        ax.set_title("Evaluated quantity: " + quantity)
        ax.set_xlabel("RMSE divided by STD of reference results")
        ax.set_ylabel("time / ms")
        ax.set_xscale("log")
        offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
        annotations_transform = ax.transData + offset
        for p, marker in zip(unique_ps, markerlist):
            selection = results_all.loc[
                (
                    (results_all["p"] == p)
                    & (results_all["quantity"] == quantity)
                )
            ]
            cutoffs = selection.level_zero_cutoff
            errors = selection.relative_rmse
            times = selection.time * 10**3
            graph = ax.plot(
                errors,
                times,
                marker=marker,
                markerfacecolor="none",
                label=f"p = {p}, varying cutoffs",
            )
            for r_cut, err, t in zip(cutoffs, errors, times):
                xy = (err, t)
                ax.annotate(
                    str(r_cut),
                    xy,
                    xytext=xy,
                    textcoords=annotations_transform,
                    ha=ha,
                    va=va,
                    color=graph[0].get_color(),
                )

        ax.set_ylim(0.0, ax.get_ylim()[1])
        ax.legend()

        filename_without_suffix = f"cost_vs_accuracy_{quantity}"
        for suffix in [".png", ".pdf"]:
            outfile = baseoutdir / (filename_without_suffix + suffix)
            print(f"- Saving plot to: {outfile}")
            fig.savefig(outfile)
