"""Tests for implementation of evaluation of short-range part.

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

from pathlib import Path
from typing import Callable, List

import jax
import jax.config
import numpy as onp
import pytest
from ase.atoms import Atoms

from msmjax.benchmark_tools import path_input_structures
from msmjax.shortrange import gen_supercell

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def fixture_dir_structures() -> Path:
    """Get directory where pre-generated structures are located."""
    # TODO: Using the structures from "data/benchmark" for tests as well is not
    #  very consistent
    return path_input_structures


@pytest.fixture(scope="module")
def fixture_structure(fixture_dir_structures) -> dict:
    """Get one pre-generated structure with a given number of particles."""
    structsfile = fixture_dir_structures / ("structures_500.npz")
    structures = onp.load(structsfile)
    return {
        "cell": structures["cells"][0],
        "positions": structures["positions"][0],
        "charges": structures["charges"][0],
    }


@pytest.fixture(scope="module")
def fixture_structure_nonortho(fixture_structure) -> dict:
    atoms = Atoms(
        positions=fixture_structure["positions"],
        charges=fixture_structure["charges"],
        cell=fixture_structure["cell"],
    )
    new_lengths = onp.diag(fixture_structure["cell"]) * (0.8, 1.0, 1.25)
    new_angles = [75, 90, 120]
    nonortho_cell = onp.concatenate([new_lengths, new_angles])
    atoms.set_cell(nonortho_cell, scale_atoms=True)
    return {
        "cell": atoms.get_cell()[...],
        "positions": atoms.get_positions(),
        "charges": atoms.get_initial_charges(),
    }


@pytest.mark.parametrize(
    "supercell_diag", [1, 2, (1, 1, 1), (2, 2, 2), (1, 2, 3)]
)
def test_gen_supercell(fixture_structure, supercell_diag):
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    super_pos, super_chg, super_cell = gen_supercell(
        positions=pos, charges=chg, cell=cell, supercell_diag=supercell_diag
    )
    atoms = Atoms(positions=pos, charges=chg, cell=cell)
    atoms = atoms.repeat(supercell_diag)

    assert onp.allclose(super_pos, atoms.get_positions())
    assert onp.allclose(super_chg, atoms.get_initial_charges())
    assert onp.allclose(super_cell, atoms.cell[...])


# def test_gen_supercell_nonortho():
#     raise ValueError


# def test_gen_supercell_2d():
#     raise ValueError
