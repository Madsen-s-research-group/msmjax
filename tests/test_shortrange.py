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
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
    compute_distance_vectors,
    gen_supercell,
    make_compute_U0,
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
    """Test cell replication against `ase.atoms.Atoms.repeat()`, 2-d case"""
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
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "pbc",
    [
        (True, True, True),
        (True, True, False),
        (True, False, False),
        (False, False, False),
        pytest.param(
            (False, True, False),
            marks=pytest.mark.xfail(
                reason="possibly misunderstanding about mixed PBCs in ase?"  # TODO
            ),
        ),
    ],
)
def test_compute_distance_vectors(fixture_structure, pbc):
    """Test correct pair distance computation under minimum image convention

    As a reference to compare to, the result from ase's `get_distances` method
    is used.
    """
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    max_cutoff = get_max_cutoff_3d(cell)
    (i, j) = jnp.triu_indices(pos.shape[0], k=1)

    deltas = compute_distance_vectors(
        positions=pos,
        cell=cell,
        pair_indices=(i, j),
        pbc=jnp.array(pbc),
    )
    atoms = Atoms(positions=pos, charges=chg, cell=cell, pbc=pbc)
    deltas_ref = atoms.get_distances(i, j, mic=True, vector=True)

    distances_ref = onp.linalg.norm(deltas_ref, axis=1)
    is_within_cutoff = distances_ref < max_cutoff
    deltas_within_cutoff = deltas[is_within_cutoff]
    deltas_within_cutoff_ase = deltas_ref[is_within_cutoff]

    assert onp.allclose(deltas_within_cutoff, deltas_within_cutoff_ase)


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
    """Test equal energy with and without supercell

    For cutoffs small enough to fit, the following should be the same:
        - Energy computed in the original cell
        - Energy computed in a supercell, but using only the particles in the
          original cell as centers
    """
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
def test_pair_term_supercell_correct_multiple(
    fixture_structure, supercell_diag
):
    """Test energy of whole supercell is right multiple of original cell's"""
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    pbc = (True, True, True)
    kernel_fn = partial(
        shortrange_quadratic_potential, r_cut=(0.99 * get_max_cutoff_3d(cell))
    )
    pair_term_fn = make_pair_term_fn(kernel_fn=kernel_fn, pbc=pbc)
    super_pos, super_chg, super_cell = gen_supercell(
        pos, chg, cell, supercell_diag
    )
    energy = pair_term_fn(pos, chg, cell)
    energy_supercell = pair_term_fn(super_pos, super_chg, super_cell)
    assert onp.isclose(energy_supercell, onp.prod(supercell_diag) * energy)


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "cutoff_multiplier, supercell_diag", [(0.99, 1), (1.99, 2)]
)
def test_pair_term_periodic_wrap_vs_replicate(
    fixture_structure, cutoff_multiplier, supercell_diag
):
    """Test periodic wrapping vs. explicit system replication

    The following should be the same in a periodic system:
        - Energy computed in the original cell, accounting for periodicity
          by wrapping around the edges
        - Energy computed in a cell replicated around the original cell on
          all sides, using only the particles in the original cell as centers
          (essentially faking periodicity in a larger non-periodic system)
    """
    # TODO: can this test be written more compactly?
    n_particles = 100
    pos = fixture_structure["positions"][:n_particles]
    chg = fixture_structure["charges"][:n_particles]
    cell = fixture_structure["cell"]

    kernel_fn = partial(
        shortrange_quadratic_potential,
        r_cut=cutoff_multiplier * get_max_cutoff_3d(cell),
    )

    n_repeats_explicit = (3, 3, 3)
    M = onp.prod(n_repeats_explicit)
    pair_term_fn_explicit_replicate = make_pair_term_fn_with_neighbor_list(
        kernel_fn=kernel_fn, pbc=(False, False, False)
    )
    pos_extended, chg_extended, cell_extended = gen_supercell(
        positions=pos,
        charges=chg,
        cell=cell,
        supercell_diag=n_repeats_explicit,
    )
    pos_extended = jnp.roll(pos_extended, (M // 2 + 1) * n_particles, axis=0)
    n_centers = pos.shape[0]
    n_total = pos_extended.shape[0]
    pair_inds_explicit_replicate = onp.where(
        onp.arange(n_centers)[:, onp.newaxis] < onp.arange(n_total)
    )
    pair_weights_explicit_replicate = onp.where(
        pair_inds_explicit_replicate[1] < n_centers, 1.0, 0.5
    )
    energy_explicit_replicate = pair_term_fn_explicit_replicate(
        pos_extended,
        chg_extended,
        cell_extended,
        neighbor_list=pair_inds_explicit_replicate,
        weights=pair_weights_explicit_replicate,
    )

    pair_term_fn_wrap = make_pair_term_fn(
        kernel_fn=kernel_fn,
        pbc=(True, True, True),
        supercell_diag=supercell_diag,
    )
    energy_wrap = pair_term_fn_wrap(pos, chg, cell)

    assert onp.isclose(energy_explicit_replicate, energy_wrap)


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "pbc", [(True, True, True), (False, False, False), (True, False, True)]
)
def test_U0_pair_term(fixture_structure, pbc):
    """Test the pair term contribution to U0

    To do this, all higher-level kernels are set to return a constant value of
    zero. In this case, U0 should equal the result of the pair term alone.
    """
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    max_cutoff = get_max_cutoff_3d(cell)
    k_0 = partial(shortrange_quadratic_potential, r_cut=0.99 * max_cutoff)
    kernel_fns = [k_0] + [lambda x: 0.0] * 2
    compute_U0 = make_compute_U0(kernel_fns=kernel_fns, pbc=pbc)
    compute_pair_term = make_pair_term_fn(kernel_fn=k_0, pbc=pbc)
    assert onp.isclose(
        compute_U0(pos, chg, cell), compute_pair_term(pos, chg, cell)
    )


@pytest.mark.parametrize(
    "fixture_structure",
    ["fixture_structure_cubic", "fixture_structure_nonortho"],
    indirect=True,
)
@pytest.mark.parametrize(
    "pbc", [(True, True, True), (False, False, False), (True, False, True)]
)
def test_U0_self_interaction_term(fixture_structure, pbc):
    """Test the self interaction term contribution to U0

    To do this, the level-zero kernel is defined to be constantly zero, and the
    higher-level kernels to be constantly one, such that the expected value
    for U0 can be calculated from the charges alone.
    """
    pos = fixture_structure["positions"]
    chg = fixture_structure["charges"]
    cell = fixture_structure["cell"]
    k_0 = lambda x: 0.0
    ks_higher = [lambda x: 1.0] * 2
    kernel_fns = [k_0] + ks_higher
    compute_U0 = make_compute_U0(kernel_fns=kernel_fns, pbc=pbc)
    ref = -len(ks_higher) * 0.5 * (chg * chg).sum()
    assert onp.isclose(compute_U0(pos, chg, cell), ref)
