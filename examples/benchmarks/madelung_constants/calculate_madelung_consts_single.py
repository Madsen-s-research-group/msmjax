# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

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

# %%
INDIR_STRUCTURES = Path("input_structures/")
PBC = (True,) * 3

# %%
# TODO: loop over compound, or turn into command-line arg
# TODO: Names:
#  Should "NaCl" mention "cubic" or "conventional" or similar?
#  "ZnS-zincblende", "ZnS-wurtzite" or just "zincblende", "wurtzite"?
#  "CaF2" or "fluorite"?
#  Nicer names for plot labels (subscript in chemical formula, no hyphen)?

# STRUCTURE = "NaCl-conventional"
# STRUCTURE = "NaCl-primitive"
STRUCTURE = "CsCl"
# STRUCTURE = "ZnS-zincblende"
# STRUCTURE = "ZnS-wurtzite"
# STRUCTURE = "CaF2"

# TODO: turn into command-line arg
BASEOUTDIR = Path("out_single/")

# %%
outdir = BASEOUTDIR / STRUCTURE
outdir.mkdir(exist_ok=True, parents=True)

structurefile = INDIR_STRUCTURES / STRUCTURES_INFO[STRUCTURE]["structurefile"]
cation_symbol = STRUCTURES_INFO[STRUCTURE]["cation_symbol"]
anion_symbol = STRUCTURES_INFO[STRUCTURE]["anion_symbol"]
cation_charge = STRUCTURES_INFO[STRUCTURE]["cation_charge"]
anion_charge = STRUCTURES_INFO[STRUCTURE]["anion_charge"]
madelung_const_target = STRUCTURES_INFO[STRUCTURE]["target_value"]

# %%
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
print(f"- Selected structure: {STRUCTURE}.")
print(f"- Reference value for Madelung constant: M = {madelung_const_target}.")
print(f"- The shortest cation-anion distance is {d_min_cation_anion:.2f} Å.")
print("#" * 72)
print()

# %%
RANGE_OF_CUTOFFS = (
    onp.concatenate([onp.arange(2, 3.61, 0.2), onp.arange(3.6, 5.21, 0.4)])
    * d_min_cation_anion
)

# Try starting with a grid spacing equal to the minimum cation-anion distance.
# But if this is more than half the (smallest) lattice constant, use half the
# smallest lattice constant instead, as otherwise there would not even be a
# single grid level.
side_lengths = onp.linalg.norm(cell, axis=1)
base_grid_spacing = min(d_min_cation_anion, 0.5 * side_lengths.min())

# %%
list_of_spacings = [
    base_grid_spacing,
    base_grid_spacing,
    0.5 * base_grid_spacing,
    0.5 * base_grid_spacing,
]
list_of_ps = [4, 6, 6, 8]
list_of_resultsfiles = []

for h, p in zip(list_of_spacings, list_of_ps):
    print(f"- Tentative grid spacing: h/d_min = {h / d_min_cation_anion:.2f}")
    # TODO: Explain why this is done:
    tmp_msm_params = set_up_msm_params(
        cell=cell,
        level_one_spacings=h,
        level_zero_cutoff=RANGE_OF_CUTOFFS[0],  # Does not matter here
        p=p,
        pbc=PBC,
        cell_mode=STRUCTURES_INFO[STRUCTURE]["cell_mode"],
        supercell_diag=None,  # Does not matter here
        dynamic_cell=False,
    )
    # TODO: one value per direction?
    actual_h = tmp_msm_params.grid_spacings[1][0]
    if tmp_msm_params.grids_defined_on_unitcube:
        # TODO: introduce `scaled_grid_spacings` to do away with the need for this?
        # TODO: one value per direction?
        actual_h *= side_lengths[0]
    print(
        f"- Actual spacing after adjusting for pbcs: "
        f"h/d_min = {actual_h / d_min_cation_anion:.2f}"
    )
    print(f"- {p=}:")

    madelung_consts_calculated = []
    highest_level_cutoffs = []

    for level_zero_cutoff in (progressbar := tqdm(RANGE_OF_CUTOFFS)):
        progressbar.set_postfix_str(
            f"r_cut_0/d_min = {level_zero_cutoff / d_min_cation_anion:.2f}"
        )

        # TODO: Use the suggest_supercell_diag currently used in
        #  calculate_madelung_consts_batch.py here as well?
        # Determine the necessary cell replication for the current cutoff.
        # First for an orthorhombic cell:
        supercell_diag = onp.ceil(
            level_zero_cutoff / (0.5 * side_lengths)
        ).astype(int)
        # To also work for the triclinic cells that are included,
        # enlarge it more:
        if STRUCTURES_INFO[STRUCTURE]["cell_mode"] == "triclinic":
            supercell_diag += 2

        msm_params = set_up_msm_params(
            cell=cell,
            level_one_spacings=actual_h,
            level_zero_cutoff=level_zero_cutoff,
            p=p,
            pbc=PBC,
            cell_mode=STRUCTURES_INFO[STRUCTURE]["cell_mode"],
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
            "compound": [STRUCTURE] * len(RANGE_OF_CUTOFFS),
            # TODO: save multiple spacing values in triclinic cases?
            "level_one_gridspacing": onp.full_like(RANGE_OF_CUTOFFS, actual_h),
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
    outfile = outdir / ("results_" + f"h-{actual_h:.2f}_p-{p}" + ".csv")
    print(f"- Saving results to {outfile}.")
    results.to_csv(outfile, index=False)
    list_of_resultsfiles.append(outfile)
    print()

# %%
fig, ax = plt.subplots()
ax.set_title(STRUCTURES_INFO[STRUCTURE]["nice_label"])
ax.set_xlabel(r"$r_{\text{cut}}^{(0)}$ / $d_{\text{min}}$")
ax.set_ylabel(r"$M - M_{\text{ref}}$")

linthresh = 1.0

# TODO: different marker styles
# TODO: Second x-axis that shows the largest effective cutoff in the system?

for resultsfile in list_of_resultsfiles:
    loaded = pd.read_csv(resultsfile)

    h = loaded["level_one_gridspacing"].values[0]
    p = loaded["p"].values[0]
    d_min_cation_anion = loaded["d_min"].values[0]

    cutoffs = loaded["level_zero_cutoff"].values
    madelung_values = loaded["madelung_value"].values

    cutoffs_relative = cutoffs / d_min_cation_anion
    target_value = STRUCTURES_INFO[loaded["compound"].values[0]][
        "target_value"
    ]
    deviations = madelung_values - target_value

    linthresh = min(linthresh, min(onp.abs(deviations)))

    label = f"$h_1 = {h / d_min_cation_anion:.2f} \, " + r"d_{\text{min}}$,"
    label += " " + f"$p = {p}$"

    ax.plot(cutoffs_relative, deviations, marker="x", label=label)
    ax.axhline(0.0, color="black", zorder=-10)

ax.set_yscale("symlog", linthresh=linthresh)

ax.legend()
plt.show()  # TODO: take out

for suffix in ["png", "pdf"]:
    outfile_plot = outdir / ("madelung_const_vs_cutoff_symlog" + "." + suffix)
    print(f"- Saving plot to {outfile_plot}")
    fig.savefig(outfile_plot)

# %%
fig, ax = plt.subplots()
ax.set_title(STRUCTURES_INFO[STRUCTURE]["nice_label"])
ax.set_xlabel(r"$r_{\text{cut}}^{(0)}$ / $d_{\text{min}}$")
ax.set_ylabel(r"$|M - M_{\text{ref}}|$")

# TODO: different marker styles
# TODO: Second x-axis that shows the largest effective cutoff in the system?

for resultsfile in list_of_resultsfiles:
    loaded = pd.read_csv(resultsfile)

    h = loaded["level_one_gridspacing"].values[0]
    p = loaded["p"].values[0]
    d_min_cation_anion = loaded["d_min"].values[0]

    cutoffs = loaded["level_zero_cutoff"].values
    madelung_values = loaded["madelung_value"].values

    cutoffs_relative = cutoffs / d_min_cation_anion
    target_value = STRUCTURES_INFO[loaded["compound"].values[0]][
        "target_value"
    ]
    deviations = madelung_values - target_value

    linthresh = min(linthresh, min(onp.abs(deviations)))

    label = f"$h_1 = {h / d_min_cation_anion:.2f} \, " + r"d_{\text{min}}$,"
    label += " " + f"$p = {p}$"

    ax.plot(cutoffs_relative, onp.abs(deviations), marker="x", label=label)

ax.set_yscale("log")

ax.legend()
plt.show()  # TODO: take out

for suffix in ["png", "pdf"]:
    outfile_plot = outdir / ("madelung_const_vs_cutoff_log" + "." + suffix)
    print(f"- Saving plot to {outfile_plot}")
    fig.savefig(outfile_plot)
