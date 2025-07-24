"""Script to calculate Madelung constants with MSM in single-structure mode.

I.e., the evaluation function is set up and compiled specifically for the one
structure it is going to be evaluated on.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

from argparse import ArgumentParser
from pathlib import Path

import ase.io
import jax
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
from tqdm import tqdm
from utils_madelung import (
    STRUCTURES_INFO,
    compute_madelung_constant,
    construct_charges_array,
    find_min_cation_anion_distance,
)

from msmjax.calculators import create_msm, set_up_msm_params

INDIR_STRUCTURES = Path("input_structures/")
PBC = (True,) * 3

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Calculate Madelung constant with MSM with different "
        "parameter settings for a single structure."
    )
    parser.add_argument(
        "--structure",
        type=str,
        required=True,
        choices=list(STRUCTURES_INFO.keys()),
        help="Name of structure for which to calculate Madelung constant.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save results to. If it exists already, "
        "the script will terminate.",
    )
    parser.add_argument(
        "-p", type=int, required=True, help="Interpolation order."
    )
    args = parser.parse_args()
    structurekey = args.structure
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True)
    p = args.p

    structurefile = (
        INDIR_STRUCTURES / STRUCTURES_INFO[structurekey]["structurefile"]
    )
    cation_symbol = STRUCTURES_INFO[structurekey]["cation_symbol"]
    anion_symbol = STRUCTURES_INFO[structurekey]["anion_symbol"]
    cation_charge = STRUCTURES_INFO[structurekey]["cation_charge"]
    anion_charge = STRUCTURES_INFO[structurekey]["anion_charge"]
    madelung_const_target = STRUCTURES_INFO[structurekey]["target_value"]

    atoms = ase.io.read(structurefile)
    positions = atoms.get_positions()
    cell = atoms.get_cell()[...]
    charges = construct_charges_array(
        atoms=atoms,
        cation_symbol=cation_symbol,
        anion_symbol=anion_symbol,
        cation_charge=cation_charge,
        anion_charge=anion_charge,
    )

    d_min_cation_anion = find_min_cation_anion_distance(
        atoms=atoms,
        cation_symbol=cation_symbol,
        anion_symbol=anion_symbol,
    )

    print("#" * 72)
    print(f"- Selected structure: {structurekey}.")
    print(
        f"- Reference value for Madelung constant: M = {madelung_const_target}."
    )
    print(
        f"- The shortest cation-anion distance is {d_min_cation_anion:.2f} Å."
    )
    print("#" * 72)
    print()

    RANGE_OF_CUTOFFS = onp.arange(2, 4.01, 0.25) * d_min_cation_anion

    # Try starting with a grid spacing equal to the minimum cation-anion distance.
    # But if this is more than half the (smallest) lattice constant, use half the
    # smallest lattice constant instead, as otherwise there would not even be a
    # single grid level.
    side_lengths = onp.linalg.norm(cell, axis=1)
    base_grid_spacing = min(d_min_cation_anion, 0.5 * side_lengths.min())

    list_of_spacings = [
        base_grid_spacing,
        0.5 * base_grid_spacing,
        0.25 * base_grid_spacing,
    ]
    list_of_result_file_names = [
        "results_base_spacing.csv",
        "results_half_base_spacing.csv",
        "results_quarter_base_spacing.csv",
    ]
    list_of_resultsfiles = []

    for h, result_file_name in zip(
        list_of_spacings, list_of_result_file_names
    ):
        print(
            f"- Tentative grid spacing: h/d_min = {h / d_min_cation_anion:.2f}"
        )
        # Set up a temporary MSM parameters object in order to obtain the
        # actual (pbc-adjusted) grid spacings, so we can print them:
        tmp_msm_params = set_up_msm_params(
            cell=cell,
            level_one_spacings=h,
            level_zero_cutoff=RANGE_OF_CUTOFFS[0],  # does not matter for this
            p=4,  # does not matter for this
            pbc=PBC,
            cell_mode=STRUCTURES_INFO[structurekey]["cell_mode"],
            supercell_diag=None,  # does not matter for this
            dynamic_cell=False,
        )
        actual_spacings = tmp_msm_params.grid_spacings[1]
        if tmp_msm_params.grids_defined_on_unitcube:
            # TODO: introduce `scaled_grid_spacings` to do away with the need for this?
            actual_spacings *= side_lengths
        formatted_scaled_spacings = ", ".join(
            [f"{x:.2f}" for x in actual_spacings / d_min_cation_anion]
        )
        print(
            f"- Actual spacings (along each direction) after adjusting for "
            f"pbcs: h/d_min = ({formatted_scaled_spacings})"
        )

        madelung_consts_calculated = []
        highest_level_cutoffs = []

        print("- Running for different cutoffs:")
        for level_zero_cutoff in (progressbar := tqdm(RANGE_OF_CUTOFFS)):
            progressbar.set_postfix_str(
                f"r_cut_0/d_min = {level_zero_cutoff / d_min_cation_anion:.2f}"
            )

            # Determine the necessary cell replication for the current cutoff.
            # First for an orthorhombic cell:
            supercell_diag = onp.ceil(
                level_zero_cutoff / (0.5 * side_lengths)
            ).astype(int)
            # To also work for the triclinic cells that are included,
            # enlarge it more:
            if STRUCTURES_INFO[structurekey]["cell_mode"] == "triclinic":
                supercell_diag += 2

            msm_params = set_up_msm_params(
                cell=cell,
                level_one_spacings=actual_spacings,
                level_zero_cutoff=level_zero_cutoff,
                p=p,
                pbc=PBC,
                cell_mode=STRUCTURES_INFO[structurekey]["cell_mode"],
                supercell_diag=supercell_diag,
                dynamic_cell=False,
            )
            evaluation_fns = create_msm(msm_params)
            energy_msm = jax.jit(evaluation_fns["energy"])(positions, charges)
            m = compute_madelung_constant(
                energy=energy_msm,
                atoms=atoms,
                cation_charge=cation_charge,
                anion_charge=anion_charge,
                cation_symbol=cation_symbol,
                anion_symbol=anion_symbol,
            )

            progressbar.set_postfix_str(
                f"r_cut_0/d_min = {level_zero_cutoff / d_min_cation_anion:.2f}, "
                f"M = {m}"
            )

            highest_level_cutoffs.append(
                msm_params.cutoffs[msm_params.max_grid_level]
            )
            madelung_consts_calculated.append(m)

        results = pd.DataFrame(
            data={
                "structure": [structurekey] * len(RANGE_OF_CUTOFFS),
                "level_one_gridspacing_a": actual_spacings[0],
                "level_one_gridspacing_b": actual_spacings[1],
                "level_one_gridspacing_c": actual_spacings[2],
                "p": onp.full_like(RANGE_OF_CUTOFFS, p, dtype=int),
                "max_grid_level": onp.full_like(
                    RANGE_OF_CUTOFFS, msm_params.max_grid_level, dtype=int
                ),
                "d_min": onp.full_like(RANGE_OF_CUTOFFS, d_min_cation_anion),
                "level_zero_cutoff": RANGE_OF_CUTOFFS,
                "highest_cutoff": highest_level_cutoffs,
                "madelung_value": madelung_consts_calculated,
            }
        )
        outfile = outdir / result_file_name
        print(f"- Saving results to {outfile}.")
        results.to_csv(outfile, index=False)
        list_of_resultsfiles.append(outfile)
        print()

    # TODO: Is setting the figsize needed at all if I don't include suptitle?
    FIGSIZE = (6.4, 4.2)  # TODO
    LINEWIDTH_TWIN_AXES = 1.5
    plt.rcParams["font.size"] = 10  # TODO

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    fig.suptitle(STRUCTURES_INFO[structurekey]["nice_label"])
    ax.set_xlabel(r"$r_{\text{cut}}^{(0)}$ / $d_{\text{min}}$")
    ax.set_ylabel(r"$|M - M_{\text{ref}}|$")

    twin1 = ax.twiny()
    twin1.tick_params(axis="x", width=LINEWIDTH_TWIN_AXES, pad=-1)
    twin2 = ax.twiny()
    twin2.spines["top"].set_position(("axes", 1.08))
    twin2.tick_params(axis="x", width=LINEWIDTH_TWIN_AXES, pad=-1)
    twin3 = ax.twiny()
    twin3.spines["top"].set_position(("axes", 1.16))
    twin3.tick_params(axis="x", width=LINEWIDTH_TWIN_AXES, pad=-1)
    twin3.set_xlabel(r"largest included cutoff / $d_{\text{min}}$")
    axes_twin = [twin1, twin2, twin3]

    list_of_markers = ["x", "o", "s"]
    for resultsfile, ax_twin, marker in zip(
        list_of_resultsfiles, axes_twin, list_of_markers
    ):
        loaded = pd.read_csv(resultsfile)
        h_1_a = loaded["level_one_gridspacing_a"].values[0]
        h_1_b = loaded["level_one_gridspacing_b"].values[0]
        h_1_c = loaded["level_one_gridspacing_c"].values[0]
        p = loaded["p"].values[0]
        d_min_cation_anion = loaded["d_min"].values[0]
        n_levels = loaded["max_grid_level"].values[0]
        cutoffs = loaded["level_zero_cutoff"].values
        cutoffs_at_max_level = loaded["highest_cutoff"]
        madelung_values = loaded["madelung_value"].values
        cutoffs_relative = cutoffs / d_min_cation_anion
        cutoffs_at_max_level_relative = (
            cutoffs_at_max_level / d_min_cation_anion
        )
        target_value = STRUCTURES_INFO[loaded["structure"].values[0]][
            "target_value"
        ]
        deviations = madelung_values - target_value
        formatted_scaled_spacings = ", ".join(
            [f"{x / d_min_cation_anion:.2f}" for x in (h_1_a, h_1_b, h_1_c)]
        )
        label = (
            f"$h_1 = ({formatted_scaled_spacings}) \, " + r"d_{\text{min}}$"
        )
        label += " $\Longrightarrow$ " + f"{n_levels} grid level(s)"
        (graph,) = ax.plot(
            cutoffs_relative,
            onp.abs(deviations),
            marker=marker,
            label=label,
            markerfacecolor="none",
        )
        ax_twin.scatter(
            cutoffs_at_max_level_relative, onp.abs(deviations), marker="none"
        )
        ax_twin.spines["top"].set_color(graph.get_color())
        ax_twin.spines["top"].set_linewidth(LINEWIDTH_TWIN_AXES)
        ax_twin.tick_params(axis="x", colors=graph.get_color())
    ax.set_yscale("log")
    ax.legend()

    for suffix in ["png", "pdf"]:
        outfile_plot = outdir / (
            "madelung_const_" + structurekey + "_vs_cutoff" + "." + suffix
        )
        print(f"- Saving plot to {outfile_plot}")
        fig.savefig(outfile_plot)
