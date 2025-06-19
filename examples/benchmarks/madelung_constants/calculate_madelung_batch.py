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

from msmjax.calculators import (
    check_cutoffs_and_spacings,
    create_msm,
    set_up_msm_params,
)
from msmjax.kernels import determine_min_kernel_stencil_size

# %%
INDIR_STRUCTURES = Path("input_structures/")
PBC = (True,) * 3

# %%
# TODO: turn into command-line arg
OUTDIR = Path("out_batch/")
OUTDIR.mkdir(exist_ok=True, parents=True)

# %%
n_dim = 3
rng = onp.random.default_rng(92398)

all_structures = list(STRUCTURES_INFO.keys())
all_atoms = [
    ase.io.read(INDIR_STRUCTURES / STRUCTURES_INFO[structure]["structurefile"])
    for structure in all_structures
]
all_numbers_of_atoms = onp.array([len(at) for at in all_atoms])
max_n_atoms = max(all_numbers_of_atoms)
min_n_atoms = min(all_numbers_of_atoms)

print(
    f"- The maximum number of atoms among all structures is {max_n_atoms} for: "
    + ", ".join(onp.array(all_structures)[all_numbers_of_atoms == max_n_atoms])
    + "."
)
print(
    f"- The minimum number of atoms among all structures is {min_n_atoms} for: "
    + ", ".join(onp.array(all_structures)[all_numbers_of_atoms == min_n_atoms])
    + "."
)
print(
    f"- => Padding all structures to the common max size of {max_n_atoms} "
    "with zero-charge ghost atoms."
)
print()

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

    padding_size = max_n_atoms - pos.shape[0]
    pos_dummy = (
        rng.uniform(low=0.0, high=1.0, size=(padding_size, n_dim)) @ cll
    )
    pos_padded = onp.concatenate([pos, pos_dummy])
    chg_padded = onp.pad(chg, (0, padding_size), constant_values=0.0)

    all_positions_padded.append(pos_padded)
    all_charges_padded.append(chg_padded)
    all_cells.append(cll)

all_positions_padded = onp.array(all_positions_padded)
all_charges_padded = onp.array(all_charges_padded)
all_cells = onp.array(all_cells)

# %%
all_d_min = []
for structure, atoms in zip(all_structures, all_atoms):
    d = find_min_cation_anion_distance(
        atoms,
        cation_symbol=STRUCTURES_INFO[structure]["cation_symbol"],
        anion_symbol=STRUCTURES_INFO[structure]["anion_symbol"],
    )
    all_d_min.append(d)
all_d_min = onp.array(all_d_min)
d_min_global = min(all_d_min)

print(
    f"- The shortest cation-anion distance among all structures is "
    + f"{min(all_d_min):.2f} Å for: "
    + ", ".join(onp.array(all_structures)[all_d_min == d_min_global])
    + "."
)
print()


# %%
def suggest_supercell_diag(cell, cutoff):
    # TODO: Put into utils?
    # TODO: Explain what is being done here (Why * 2? Why + 1?)
    side_lengths = onp.linalg.norm(cell, axis=1)
    supercell_diag = determine_min_kernel_stencil_size(
        cell=cell, spacings=side_lengths, cutoff=2 * cutoff
    )
    supercell_diag = tuple(s + 1 for s in supercell_diag)
    return supercell_diag


# %%
# TODO: somewhat arbitrary choices
# TODO: print n_divisions or number of grid points somewhere?
n_divisions = 4
reference_cell = all_cells[0]
reference_side_lengths = onp.linalg.norm(reference_cell, axis=1)
reference_level_one_spacings = reference_side_lengths / n_divisions

# TODO: loop over cutoff values?
LEVEL_ZERO_CUTOFF = 5.0

print(
    "- Determining the necessary supercell size to accommodate the "
    "short-range cutoff for all cells to be evaluated."
)
supercell_diags_all_structures = []
for cell in all_cells:
    supercell_diags_all_structures.append(
        suggest_supercell_diag(cell=cell, cutoff=LEVEL_ZERO_CUTOFF)
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
    side_lengths = onp.linalg.norm(cell, axis=1)
    spacings = side_lengths / n_divisions
    stencil_extents = determine_min_kernel_stencil_size(
        cell=cell, spacings=spacings, cutoff=2 * LEVEL_ZERO_CUTOFF
    )
    stencil_extents_intermed_all_structures.append(stencil_extents)
common_stencil_extents_intermed = tuple(
    onp.array(stencil_extents_intermed_all_structures).max(axis=0).tolist()
)
print(f"- Found {common_stencil_extents_intermed}.")

# %%
msm_params = set_up_msm_params(
    cell=reference_cell,
    level_one_spacings=reference_level_one_spacings,
    level_zero_cutoff=LEVEL_ZERO_CUTOFF,
    pbc=PBC,
    cell_mode="triclinic",
    dynamic_cell=True,
    supercell_diag=common_supercell_diag,
    intermediate_kernel_stencil_extents=common_stencil_extents_intermed,
)
evaluation_fns = create_msm(msm_params)
calc_energy_batch = jax.jit(jax.vmap(evaluation_fns["energy"]))
# TODO
# all_energies_msm = calc_energy_batch(
#     all_positions_padded, all_charges_padded, all_cells
# )

# %%
# It's generally a good idea to (re-)check that the cutoffs fit and the
# spacings are reasonable for all cell shapes to be evaluated:
print(
    "- With the given settings, the actual level-one spacings for all "
    "structures are:"
)
for structure, cell, d_min in zip(all_structures, all_cells, all_d_min):
    actual_spacings = check_cutoffs_and_spacings(cell, msm_params)
    print(f"  {structure + ':':<18} h_1/d_min = {actual_spacings / d_min}")

# %%
all_energies_msm = calc_energy_batch(
    all_positions_padded, all_charges_padded, all_cells
)

deviations = []
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
    deviations.append(m_calculated - m_ref)

# %%
# TODO: make plot
