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
from functools import partial
from pathlib import Path
from typing import Callable, List

import jax
import jax.config
import jax.numpy as jnp
import numpy as onp
import pytest
from ase.atoms import Atoms

from msmjax.benchmark_tools import path_input_structures
from msmjax.shortrange import (
    gen_supercell,
    make_pair_term_fn,
    make_pair_term_fn_with_neighbor_list,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def fixture_dir_structures() -> Path:
    """Get directory where pre-generated structures are located."""
    # TODO: Using the structures from "data/benchmark" for tests as well is not
    #  very consistent
    return path_input_structures


@pytest.fixture(scope="module")
def fixture_structure_cubic(fixture_dir_structures) -> dict:
    """Get a cubic structure by loading from pre-generated ones"""
    structsfile = fixture_dir_structures / ("structures_500.npz")
    structures = onp.load(structsfile)
    return {
        "cell": structures["cells"][0].astype(onp.float64),
        "positions": structures["positions"][0].astype(onp.float64),
        "charges": structures["charges"][0].astype(onp.float64),
    }


@pytest.fixture(scope="module")
def fixture_structure_nonortho(fixture_structure_cubic) -> dict:
    """Get a non-orthorhombic structure by stretching/distorting a cubic one

    Args:
        fixture_structure_cubic: The original cubic structure

    Returns:
        Stretched/distorted version of the original cubic structure
    """
    atoms = Atoms(
        positions=fixture_structure_cubic["positions"],
        charges=fixture_structure_cubic["charges"],
        cell=fixture_structure_cubic["cell"],
    )
    new_lengths = onp.diag(fixture_structure_cubic["cell"]) * (0.8, 1.0, 1.25)
    new_angles = [75, 90, 120]
    nonortho_cell = onp.concatenate([new_lengths, new_angles])
    atoms.set_cell(nonortho_cell, scale_atoms=True)
    return {
        "cell": atoms.get_cell()[...],
        "positions": atoms.get_positions(),
        "charges": atoms.get_initial_charges(),
    }


@pytest.fixture(scope="module")
def fixture_structure(request):
    """Helper fixture for requesting a specific structure fixture"""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "supercell_diag", [1, 2, (1, 1, 1), (2, 2, 2), (1, 2, 3)]
)
def test_gen_supercell(fixture_structure, supercell_diag):
    """Test cell replication result against `ase.atoms.Atoms.repeat()`"""
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


@pytest.mark.parametrize("supercell_diag", [1, 2, (1, 1), (2, 2), (2, 3)])
def test_gen_supercell_2d(fixture_structure_cubic, supercell_diag):
    pos = fixture_structure_cubic["positions"]
    chg = fixture_structure_cubic["charges"]
    cell = fixture_structure_cubic["cell"]
    pos_2d = pos[:, :2]
    cell_2d = cell[:2, :2]
    super_pos_2d, super_chg, super_cell_2d = gen_supercell(
        positions=pos_2d,
        charges=chg,
        cell=cell_2d,
        supercell_diag=supercell_diag,
    )
    atoms = Atoms(positions=pos, charges=chg, cell=cell)
    if onp.ndim(supercell_diag) == 0:
        atoms = atoms.repeat((supercell_diag, supercell_diag, 1))
    else:
        atoms = atoms.repeat(supercell_diag + (1,))

    assert onp.allclose(super_pos_2d, atoms.get_positions()[:, :2])
    assert onp.allclose(super_chg, atoms.get_initial_charges())
    assert onp.allclose(super_cell_2d, atoms.cell[:2, :2])


def shortrange_quadratic_potential(r, r_cut):
    return jnp.where(r < r_cut, (r - r_cut) ** 2, 0.0)


def shortrange_counting_potential(r, r_cut):
    return jnp.where(r < r_cut, 1.0, 0.0)


def get_max_cutoff_3d(cell: jnp.ndarray):
    """Get the maximum cutoff value that fits into a 3D cell.

    Args:
        cell: Cell, shape=(3, 3).

    Returns:
        Cutoff radius
    """
    return jnp.min(
        jnp.fabs(
            jnp.linalg.det(cell)
            / jnp.array(
                [
                    jnp.linalg.norm(jnp.cross(i, j))
                    for i, j in zip(cell, jnp.roll(cell, 1, axis=0))
                ]
            )
        )
        / 2.0
    )


@pytest.mark.parametrize(
    "pbc, supercell_diag",
    [
        ((False, False, False), (2, 2, 2)),
        ((False, False, True), (2, 2, 1)),
        ((False, False, True), (2, 1, 1)),
        ((False, True, True), (2, 3, 4)),
        ((False, True, True), (3, 1, 2)),
    ],
)
def test_error_supercell_nonperiodic(pbc, supercell_diag):
    """Test if error when cell replication along non-periodic axis requested"""
    with pytest.raises(ValueError, match=r"along non-periodic axes"):
        make_pair_term_fn(
            kernel_fn=partial(shortrange_quadratic_potential, r_cut=1.0),
            pbc=pbc,
            supercell_diag=supercell_diag,
        )


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
def test_pair_term_with_and_without_supercell(fixture_structure):
    # TODO: docstring
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    pbc = (True, True, True)
    kernel_fn = partial(
        shortrange_quadratic_potential, r_cut=(0.99 * get_max_cutoff_3d(cell))
    )
    pair_term_fn = make_pair_term_fn(kernel_fn=kernel_fn, pbc=pbc)
    pair_term_fn_supercell = make_pair_term_fn(
        kernel_fn=kernel_fn,
        pbc=pbc,
        supercell_diag=(2, 2, 2),
    )
    assert onp.isclose(
        pair_term_fn(pos, chg, cell), pair_term_fn_supercell(pos, chg, cell)
    )


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
@pytest.mark.parametrize("supercell_diag", [(2, 2, 2), (1, 2, 3)])
def test_pair_term_supercell_multiple(fixture_structure, supercell_diag):
    # TODO: docstring
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    pbc = (True, True, True)
    kernel_fn = partial(
        shortrange_quadratic_potential, r_cut=(0.99 * get_max_cutoff_3d(cell))
    )
    pair_term_fn = make_pair_term_fn_with_neighbor_list(
        kernel_fn=kernel_fn, pbc=pbc
    )
    super_pos, super_chg, super_cell = gen_supercell(
        pos, chg, cell, supercell_diag
    )
    energy = pair_term_fn(
        pos, chg, cell, neighbor_list=jnp.triu_indices(pos.shape[0], k=1)
    )
    energy_supercell = pair_term_fn(
        super_pos,
        super_chg,
        super_cell,
        neighbor_list=jnp.triu_indices(super_pos.shape[0], k=1),
    )
    assert onp.isclose(energy_supercell, onp.prod(supercell_diag) * energy)


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
def test_pair_term_periodic_wrap_vs_replicate(fixture_structure):
    n_particles = 100
    pos = fixture_structure["positions"][:n_particles]
    chg = fixture_structure["charges"][:n_particles]
    cell = fixture_structure["cell"]

    kernel_fn = partial(
        shortrange_quadratic_potential, r_cut=(0.99 * get_max_cutoff_3d(cell))
    )

    n_repeats_explicit = (3, 3, 3)
    M = onp.prod(n_repeats_explicit)
    pair_term_fn_explicit_replicate = make_pair_term_fn_with_neighbor_list(
        kernel_fn=kernel_fn, pbc=(False, False, False)
    )
    pos_ext, chg_ext, cell_ext = gen_supercell(
        positions=pos,
        charges=chg,
        cell=cell,
        supercell_diag=n_repeats_explicit,
    )
    pos_ext = jnp.roll(pos_ext, (M // 2 + 1) * n_particles, axis=0)
    n_centers = pos.shape[0]
    n_total = pos_ext.shape[0]
    pair_inds_explicit_replicate = onp.where(
        onp.arange(n_centers)[:, onp.newaxis] < onp.arange(n_total)
    )
    pair_weights_explicit_replicate = onp.where(
        pair_inds_explicit_replicate[1] < n_centers, 1.0, 0.5
    )
    energy_explicit_replicate = pair_term_fn_explicit_replicate(
        pos_ext,
        chg_ext,
        cell_ext,
        neighbor_list=pair_inds_explicit_replicate,
        weights=pair_weights_explicit_replicate,
    )

    pair_term_fn_wrap = make_pair_term_fn(
        kernel_fn=kernel_fn, pbc=(True, True, True), supercell_diag=2
    )
    energy_wrap = pair_term_fn_wrap(pos, chg, cell)

    assert onp.isclose(energy_explicit_replicate, energy_wrap)
