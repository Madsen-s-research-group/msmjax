import math

import numpy as onp

# From https://doi.org/10.1063/1.1744060
STRUCTURES_INFO = {
    "NaCl-conventional": {
        "structurefile": "NaCl_mp-22862_conventional_standard.cif",
        "cell_mode": "ortho",
        "cation_symbol": "Na",
        "anion_symbol": "Cl",
        "cation_charge": 1,
        "anion_charge": -1,
        "target_value": 1.747_564_594_6,
        "nice_label": "NaCl (conventional)",
    },
    "NaCl-primitive": {
        "structurefile": "NaCl-primitive.cif",
        "cell_mode": "triclinic",
        "cation_symbol": "Na",
        "anion_symbol": "Cl",
        "cation_charge": 1,
        "anion_charge": -1,
        "target_value": 1.747_564_594_6,
        "nice_label": "NaCl (primitive)",
    },
    "CsCl": {
        "structurefile": "CsCl_mp-22865_conventional_standard.cif",
        "cell_mode": "ortho",
        "cation_symbol": "Cs",
        "anion_symbol": "Cl",
        "cation_charge": 1,
        "anion_charge": -1,
        "target_value": 1.762_674_773_0,
        "nice_label": "CsCl",
    },
    "ZnS-zincblende": {
        "structurefile": "ZnS_mp-10695_conventional_standard.cif",
        "cell_mode": "ortho",
        "cation_symbol": "Zn",
        "anion_symbol": "S",
        "cation_charge": 2,
        "anion_charge": -2,
        "target_value": 1.638_055_053_3,
        "nice_label": "ZnS (zincblende)",
    },
    "ZnS-wurtzite": {
        "structurefile": "ZnS-wurtzite.xyz",  # TODO
        "cell_mode": "triclinic",
        "cation_symbol": "Zn",
        "anion_symbol": "S",
        "cation_charge": 2,
        "anion_charge": -2,
        "target_value": 1.641_32,
        "nice_label": "ZnS (wurtzite)",
    },
    "CaF2": {
        "structurefile": "CaF2_mp-2741_conventional_standard.cif",
        "cell_mode": "ortho",
        "cation_symbol": "Ca",
        "anion_symbol": "F",
        "cation_charge": 2,
        "anion_charge": -1,
        "target_value": 5.038_784_879_8,
        "nice_label": "CaF$_2$",
    },
}


def construct_charges_array(
    atoms, cation_symbol, anion_symbol, cation_charge, anion_charge
):
    indices_cations = atoms.symbols.indices()[cation_symbol]
    indices_anions = atoms.symbols.indices()[anion_symbol]
    charges = onp.zeros_like(atoms, dtype=float)
    charges[indices_cations] = cation_charge
    charges[indices_anions] = anion_charge
    return charges


def find_min_cation_anion_distance(atoms, cation_symbol, anion_symbol):
    indices_cations = atoms.symbols.indices()[cation_symbol]
    indices_anions = atoms.symbols.indices()[anion_symbol]
    all_distances = atoms.get_all_distances(mic=True)
    min_cation_anion_distance = onp.min(
        all_distances[indices_cations, :][:, indices_anions]
    )
    return min_cation_anion_distance


def compute_madelung_constant(
    energy,
    atoms,
    cation_charge,
    anion_charge,
    cation_symbol,
    anion_symbol,
):
    # TODO: Reference to the exact formula that I'm using?
    z = math.gcd(abs(cation_charge), abs(anion_charge))
    all_distances = atoms.get_all_distances(mic=True)
    indices_cations = atoms.symbols.indices()[cation_symbol]
    indices_anions = atoms.symbols.indices()[anion_symbol]
    min_cation_anion_distance = onp.min(
        all_distances[indices_cations, :][:, indices_anions]
    )
    energy_ref = z**2 / min_cation_anion_distance
    _, n_formula_units = atoms.symbols.formula.reduce()

    return onp.abs(energy) / (n_formula_units * energy_ref)
