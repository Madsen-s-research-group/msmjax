import os

from msmjax.benchmark_tools import calc_relative_rmse_percent

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import jax
import numpy as onp
import pytest
from matscipy.neighbours import neighbour_list

from msmjax.calculators import create_msm, set_up_msm_params_static_cell


@pytest.fixture(scope="module")
def fixture_datadir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def fixture_nonperiodic_cubic(fixture_datadir):
    # TODO: return the NpzFile instance
    #  or directly pos, chg, cell, energy_ref, forces_ref, ...?
    return onp.load(fixture_datadir / "nonperiodic_cubic.npz")


@pytest.fixture(scope="module")
def fixture_nonperiodic_ortho_diff_sides(fixture_datadir):
    # TODO: return the NpzFile instance
    #  or directly pos, chg, cell, energy_ref, forces_ref, ...?
    return onp.load(
        fixture_datadir / "nonperiodic_ortho-different-sidelengths.npz"
    )


@pytest.fixture(scope="module")
def fixture_nonperiodic_triclinic(fixture_datadir):
    # TODO: return the NpzFile instance
    #  or directly pos, chg, cell, energy_ref, forces_ref, ...?
    return onp.load(fixture_datadir / "nonperiodic_triclinic.npz")


@pytest.fixture(scope="module")
def fixture_periodic_cubic(fixture_datadir):
    # TODO: return the NpzFile instance
    #  or directly pos, chg, cell, energy_ref, forces_ref, ...?
    return onp.load(fixture_datadir / "periodic_cubic.npz")


@pytest.fixture(scope="module")
def fixture_periodic_ortho_diff_sides(fixture_datadir):
    # TODO: return the NpzFile instance
    #  or directly pos, chg, cell, energy_ref, forces_ref, ...?
    return onp.load(
        fixture_datadir / "periodic_ortho-different-sidelengths.npz"
    )


@pytest.fixture(scope="module")
def fixture_periodic_triclinic(fixture_datadir):
    # TODO: return the NpzFile instance
    #  or directly pos, chg, cell, energy_ref, forces_ref, ...?
    return onp.load(fixture_datadir / "periodic_triclinic.npz")


# TODO: Test static/dynamic cell?
# TODO: Test different grid spacings along different axes?
# TODO: Test with/without neighbor list?
# TODO: Test serialization/deserialization of MSMParams (in this module or elsewhere?)


def test_nonperiodic_cubic(fixture_nonperiodic_cubic):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "ortho"
    pbc = (False, False, False)
    # TODO: spacings and cutoff should be defined outside (fixtures?)
    level_one_spacings = 1.0
    level_zero_cutoff = 3.0

    cell = fixture_nonperiodic_cubic["cell"]
    pos = fixture_nonperiodic_cubic["positions"]
    chg = fixture_nonperiodic_cubic["charges"]
    (n_particles, n_dim) = pos.shape
    energy_ref = fixture_nonperiodic_cubic["energy"]
    forces_ref = fixture_nonperiodic_cubic["forces"]

    msm_params = set_up_msm_params_static_cell(
        cell=cell,
        cell_mode=cell_mode,
        pbc=pbc,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        n_particles=n_particles,
    )
    _, calc_forces, _, _ = create_msm(msm_params)
    forces_msm = jax.jit(calc_forces)(pos, chg)

    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm, forces_ref) < 1.0


def test_nonperiodic_ortho_diff_sides(fixture_nonperiodic_ortho_diff_sides):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "ortho"
    pbc = (False, False, False)
    # TODO: spacings and cutoff should be defined outside (fixtures?)
    level_one_spacings = 1.0
    level_zero_cutoff = 3.0

    cell = fixture_nonperiodic_ortho_diff_sides["cell"]
    pos = fixture_nonperiodic_ortho_diff_sides["positions"]
    chg = fixture_nonperiodic_ortho_diff_sides["charges"]
    (n_particles, n_dim) = pos.shape
    energy_ref = fixture_nonperiodic_ortho_diff_sides["energy"]
    forces_ref = fixture_nonperiodic_ortho_diff_sides["forces"]

    msm_params = set_up_msm_params_static_cell(
        cell=cell,
        cell_mode=cell_mode,
        pbc=pbc,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        n_particles=n_particles,
    )
    _, calc_forces, _, _ = create_msm(msm_params)
    forces_msm = jax.jit(calc_forces)(pos, chg)

    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm, forces_ref) < 1.0


def test_nonperiodic_triclinic(fixture_nonperiodic_triclinic):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "general"  # TODO
    pbc = (False, False, False)
    # TODO: spacings and cutoff should be defined outside (fixtures?)
    level_one_spacings = 1.0
    level_zero_cutoff = 3.0

    cell = fixture_nonperiodic_triclinic["cell"]
    pos = fixture_nonperiodic_triclinic["positions"]
    chg = fixture_nonperiodic_triclinic["charges"]
    (n_particles, n_dim) = pos.shape
    energy_ref = fixture_nonperiodic_triclinic["energy"]
    forces_ref = fixture_nonperiodic_triclinic["forces"]

    msm_params = set_up_msm_params_static_cell(
        cell=cell,
        cell_mode=cell_mode,
        pbc=pbc,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        n_particles=n_particles,
    )
    _, calc_forces, _, _ = create_msm(msm_params)
    forces_msm = jax.jit(calc_forces)(pos, chg)

    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm, forces_ref) < 1.0


def test_periodic_cubic(fixture_periodic_cubic):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "ortho"
    pbc = (True, True, True)
    # TODO: spacings and cutoff should be defined outside (fixtures?)
    level_one_spacings = 1.0
    level_zero_cutoff = 3.0

    cell = fixture_periodic_cubic["cell"]
    pos = fixture_periodic_cubic["positions"]
    chg = fixture_periodic_cubic["charges"]
    (n_particles, n_dim) = pos.shape
    energy_ref = fixture_periodic_cubic["energy"]
    forces_ref = fixture_periodic_cubic["forces"]

    msm_params = set_up_msm_params_static_cell(
        cell=cell,
        cell_mode=cell_mode,
        pbc=pbc,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        n_particles=n_particles,
    )
    _, calc_forces, _, _ = create_msm(msm_params)
    forces_msm = jax.jit(calc_forces)(pos, chg)

    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm, forces_ref) < 1.0


def test_periodic_ortho_diff_sides(fixture_periodic_ortho_diff_sides):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "ortho"
    pbc = (True, True, True)
    # TODO: spacings and cutoff should be defined outside (fixtures?)
    level_one_spacings = 1.0
    level_zero_cutoff = 3.0

    cell = fixture_periodic_ortho_diff_sides["cell"]
    pos = fixture_periodic_ortho_diff_sides["positions"]
    chg = fixture_periodic_ortho_diff_sides["charges"]
    (n_particles, n_dim) = pos.shape
    energy_ref = fixture_periodic_ortho_diff_sides["energy"]
    forces_ref = fixture_periodic_ortho_diff_sides["forces"]

    msm_params = set_up_msm_params_static_cell(
        cell=cell,
        cell_mode=cell_mode,
        pbc=pbc,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        n_particles=n_particles,
    )
    _, calc_forces, _, _ = create_msm(msm_params)
    forces_msm = jax.jit(calc_forces)(pos, chg)

    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm, forces_ref) < 1.0
