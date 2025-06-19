# %%
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
from utils_madelung import (
    STRUCTURES_INFO,
    compute_madelung_constant,
    construct_charges_array,
    find_min_cation_anion_distance,
)

from msmjax.calculators import (
    check_cutoffs_and_spacings,
    create_msm,
    set_up_msm_params,
)
from msmjax.kernels import determine_min_kernel_stencil_size

INDIR_STRUCTURES = Path("input_structures/")
PBC = (True,) * 3


def suggest_supercell_diag(cell, cutoff):
    # TODO: Put into utils?
    # TODO: Explain what is being done here (Why * 2? Why + 1?)
    side_lengths = onp.linalg.norm(cell, axis=1)
    supercell_diag = determine_min_kernel_stencil_size(
        cell=cell, spacings=side_lengths, cutoff=2 * cutoff
    )
    supercell_diag = tuple(s + 1 for s in supercell_diag)
    return supercell_diag


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save results to. If it exists already, "
        "the script will terminate.",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    resultsfile = outdir / "results.csv"

    all_structures = list(STRUCTURES_INFO.keys())
    all_atoms = [
        ase.io.read(
            INDIR_STRUCTURES / STRUCTURES_INFO[structure]["structurefile"]
        )
        for structure in all_structures
    ]
    all_numbers_of_atoms = onp.array([len(at) for at in all_atoms])
    max_n_atoms = max(all_numbers_of_atoms)
    min_n_atoms = min(all_numbers_of_atoms)
    structures_with_most_atoms = onp.array(all_structures)[
        all_numbers_of_atoms == max_n_atoms
    ]
    structures_with_fewest_atoms = onp.array(all_structures)[
        all_numbers_of_atoms == min_n_atoms
    ]
    print(
        f"- The maximum number of atoms among all structures is {max_n_atoms} for: "
        + ", ".join(structures_with_most_atoms)
        + "."
    )
    print(
        f"- The minimum number of atoms among all structures is {min_n_atoms} for: "
        + ", ".join(structures_with_fewest_atoms)
        + "."
    )
    print(
        f"- => Padding all structures to the common max size of {max_n_atoms} "
        "with zero-charge ghost atoms."
    )
    print()

    # TODO: explanation of random padding?
    n_dim = 3
    rng = onp.random.default_rng(92398)
    all_positions_padded = []
    all_charges_padded = []
    all_cells = []
    for structure, atoms in zip(all_structures, all_atoms):
        pos = atoms.get_positions()
        chg = construct_charges_array(
            atoms=atoms,
            cation_symbol=STRUCTURES_INFO[structure]["cation_symbol"],
            anion_symbol=STRUCTURES_INFO[structure]["anion_symbol"],
            cation_charge=STRUCTURES_INFO[structure]["cation_charge"],
            anion_charge=STRUCTURES_INFO[structure]["anion_charge"],
        )
        cll = atoms.get_cell()[...]

        # Scale all structures such that the minimum cation-anion distance is one.
        # This leaves Madelung constants unchanged, but simplifies MSM setup and
        # makes the meaning of the cutoff comparable between structures.
        d = find_min_cation_anion_distance(
            atoms,
            cation_symbol=STRUCTURES_INFO[structure]["cation_symbol"],
            anion_symbol=STRUCTURES_INFO[structure]["anion_symbol"],
        )
        pos /= d
        cll /= d
        atoms.set_cell(cll, scale_atoms=True)

        n_ghost_atoms = max_n_atoms - pos.shape[0]
        pos_dummy = (
            rng.uniform(low=0.0, high=1.0, size=(n_ghost_atoms, n_dim)) @ cll
        )
        pos_padded = onp.concatenate([pos, pos_dummy])
        chg_padded = onp.pad(chg, (0, n_ghost_atoms), constant_values=0.0)

        all_positions_padded.append(pos_padded)
        all_charges_padded.append(chg_padded)
        all_cells.append(cll)

    all_positions_padded = onp.array(all_positions_padded)
    all_charges_padded = onp.array(all_charges_padded)
    all_cells = onp.array(all_cells)

    cell_largest_structure = all_cells[onp.argmax(all_numbers_of_atoms)]
    # TODO: make clear that this is the d_min of one to which we have scaled all structures
    d_min = 1.0
    target_reference_spacing = d_min  # TODO: reduce to d_min/2?

    # TODO: explanation
    prelim_msm_params = set_up_msm_params(
        cell=cell_largest_structure,
        level_one_spacings=target_reference_spacing,
        level_zero_cutoff=1.0,  # Does not matter for this
        pbc=PBC,
        cell_mode="triclinic",
        dynamic_cell=True,
    )
    n_gridpoints = prelim_msm_params.grid_shapes[1]
    print(n_gridpoints)  # TODO
    reference_spacings = onp.linalg.norm(
        cell_largest_structure, axis=1
    ) / onp.array(n_gridpoints)
    print(reference_spacings)  # TODO

    for i, level_zero_cutoff in enumerate([2.0, 3.0, 4.0, 5.0, 6.0]):
        print("#" * 72)
        print(f"level_zero_cutoff = {level_zero_cutoff:.2f}")
        print("#" * 72)
        print(
            "- Determining the necessary supercell size to accommodate the "
            "short-range cutoff for all cells to be evaluated."
        )
        supercell_diags_all_structures = []
        for cell in all_cells:
            supercell_diags_all_structures.append(
                suggest_supercell_diag(cell=cell, cutoff=level_zero_cutoff)
            )
        common_supercell_diag = tuple(
            onp.array(supercell_diags_all_structures).max(axis=0).tolist()
        )
        print(f"- Found {common_supercell_diag}.")
        print(
            "- Determining the stencil size (measured from, and not including, the "
            "point at the center) required to accommodate the cutoffs during grid part "
            "of calculation, for all structures to be evaluated."
        )
        stencil_extents_intermed_all_structures = []
        for cell in all_cells:
            # TODO: explanation?
            side_lengths = onp.linalg.norm(cell, axis=1)
            spacings = side_lengths / onp.array(n_gridpoints)
            stencil_extents = determine_min_kernel_stencil_size(
                cell=cell, spacings=spacings, cutoff=2 * level_zero_cutoff
            )
            stencil_extents_intermed_all_structures.append(stencil_extents)
        common_stencil_extents_intermed = tuple(
            onp.array(stencil_extents_intermed_all_structures)
            .max(axis=0)
            .tolist()
        )
        print(f"- Found {common_stencil_extents_intermed}.")

        msm_params = set_up_msm_params(
            cell=cell_largest_structure,  # TODO: name "reference_spacings"?
            level_one_spacings=reference_spacings,
            level_zero_cutoff=level_zero_cutoff,
            pbc=PBC,
            cell_mode="triclinic",
            dynamic_cell=True,
            supercell_diag=common_supercell_diag,
            intermediate_kernel_stencil_extents=common_stencil_extents_intermed,
        )
        # TODO: Do not repeat this output for every cutoff value?
        print(
            "- With the given settings, the actual level-one spacings for all "
            "structures are:"
        )
        for structure, cell in zip(all_structures, all_cells):
            actual_spacings = check_cutoffs_and_spacings(cell, msm_params)
            print(
                f"  {structure + ':':<18} h_1/d_min = {actual_spacings / d_min}"
            )

        evaluation_fns = create_msm(msm_params)
        calc_energy_batch = jax.jit(jax.vmap(evaluation_fns["energy"]))
        all_energies_msm = calc_energy_batch(
            all_positions_padded, all_charges_padded, all_cells
        )
        errors = []
        for structure, energy, atoms in zip(
            all_structures, all_energies_msm, all_atoms
        ):
            m_calculated = compute_madelung_constant(
                energy=energy,
                atoms=atoms,
                cation_charge=STRUCTURES_INFO[structure]["cation_charge"],
                anion_charge=STRUCTURES_INFO[structure]["anion_charge"],
                cation_symbol=STRUCTURES_INFO[structure]["cation_symbol"],
                anion_symbol=STRUCTURES_INFO[structure]["anion_symbol"],
            )
            m_ref = STRUCTURES_INFO[structure]["target_value"]
            errors.append(m_calculated - m_ref)

        results_tmp = pd.DataFrame(
            data={
                "structure": all_structures,
                "level_zero_cutoff": onp.full(
                    len(all_structures), level_zero_cutoff
                ),
                "error": errors,
            }
        )
        print(f"- Writing results to {resultsfile}.")
        if i == 0:
            results_tmp.to_csv(resultsfile, index=False, mode="w")
        else:
            results_tmp.to_csv(
                resultsfile, index=False, mode="a", header=False
            )
        print()

    markerlist = ["v", "^", "s", "o", "d", "p"]
    results = pd.read_csv(resultsfile)
    unique_structurekeys = results["structure"].unique()
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$r_{\text{cut}}^{(0)}$ / $d_{\text{min}}$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$|M - M_{\text{ref}}|$")
    for structurekey, marker in zip(unique_structurekeys, markerlist):
        selection = results.loc[
            results.structure == structurekey,
            ["level_zero_cutoff", "error"],
        ]
        ax.plot(
            selection.level_zero_cutoff,
            onp.abs(selection.error),
            label=STRUCTURES_INFO[structurekey]["nice_label"],
            marker=marker,
            markerfacecolor="none",
            markersize=10,
        )
    ax.legend()
    for suffix in ["png", "pdf"]:
        outfile_plot = outdir / (
            "madelung_consts_batch_vs_cutoff_logscale" + "." + suffix
        )
        print(f"- Saving plot to {outfile_plot}")
        fig.savefig(outfile_plot)
