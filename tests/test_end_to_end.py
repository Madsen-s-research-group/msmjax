import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import jax
import numpy as onp
import pytest

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import calc_relative_rmse_percent

# For closeness checks to pass in single precision
ATOL = 5.0e-6
# TODO: Use fixtures for these?
LEVEL_ONE_SPACINGS = 1.0
LEVEL_ZERO_CUTOFF = 3.0


@pytest.fixture(scope="module")
def fixture_datadir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def fixture_nonperiodic_cubic(fixture_datadir):
    data = onp.load(fixture_datadir / "nonperiodic_cubic.npz")
    cell_mode = "ortho"
    pbc = (False,) * 3
    return data, cell_mode, pbc


@pytest.fixture(scope="module")
def fixture_nonperiodic_ortho_diff_sides(fixture_datadir):
    data = onp.load(
        fixture_datadir / "nonperiodic_ortho-different-sidelengths.npz"
    )
    cell_mode = "ortho"
    pbc = (False,) * 3
    return data, cell_mode, pbc


@pytest.fixture(scope="module")
def fixture_nonperiodic_triclinic(fixture_datadir):
    data = onp.load(fixture_datadir / "nonperiodic_triclinic.npz")
    cell_mode = "triclinic"
    pbc = (False,) * 3
    return data, cell_mode, pbc


@pytest.fixture(scope="module")
def fixture_periodic_cubic(fixture_datadir):
    data = onp.load(fixture_datadir / "periodic_cubic.npz")
    cell_mode = "ortho"
    pbc = (True,) * 3
    return data, cell_mode, pbc


@pytest.fixture(scope="module")
def fixture_periodic_ortho_diff_sides(fixture_datadir):
    data = onp.load(
        fixture_datadir / "periodic_ortho-different-sidelengths.npz"
    )
    cell_mode = "ortho"
    pbc = (True,) * 3
    return data, cell_mode, pbc


@pytest.fixture(scope="module")
def fixture_periodic_triclinic(fixture_datadir):
    data = onp.load(fixture_datadir / "periodic_triclinic.npz")
    cell_mode = "triclinic"
    pbc = (True,) * 3
    return data, cell_mode, pbc


# TODO: Test different grid spacings along different axes?
# TODO: Test with/without neighbor list?
# TODO: Test serialization/deserialization of MSMParams (in this module or elsewhere?)


@pytest.fixture(scope="module")
def fixture_system_definition(request):
    """Helper fixture for requesting a specific structure and setup"""
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "fixture_system_definition",
    [
        "fixture_nonperiodic_cubic",
        "fixture_nonperiodic_ortho_diff_sides",
        "fixture_nonperiodic_triclinic",
        "fixture_periodic_cubic",
        "fixture_periodic_ortho_diff_sides",
        "fixture_periodic_triclinic",
    ],
    indirect=True,
)
def test_combined(fixture_system_definition):
    data, cell_mode, pbc = fixture_system_definition
    pos = data["positions"]
    chg = data["charges"]
    cell = data["cell"]
    energy_ref = data["energy"]
    forces_ref = data["forces"]

    (n_particles, n_dim) = pos.shape

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=LEVEL_ONE_SPACINGS,
        level_zero_cutoff=LEVEL_ZERO_CUTOFF,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Add checks for energy, stress, chargegrad
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=LEVEL_ONE_SPACINGS,
        level_zero_cutoff=LEVEL_ZERO_CUTOFF,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
    )
    calc_energy, calc_forces, _, _, _ = create_msm(params_dyncell)
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    # TODO: Add checks for energy, stress, chargegrad
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)


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

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
    )
    calc_energy, calc_forces, _, _, _ = create_msm(params_dyncell)
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)


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

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
    )
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)


def test_nonperiodic_triclinic(fixture_nonperiodic_triclinic):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "triclinic"  # TODO
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

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
    )
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)


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

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
    )
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)


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

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
    )
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)


def test_periodic_triclinic(fixture_periodic_triclinic):
    # TODO: cell mode and pbc are specific to the structure fixture
    cell_mode = "triclinic"  # TODO
    pbc = (True, True, True)
    # TODO: spacings and cutoff should be defined outside (fixtures?)
    level_one_spacings = 1.0
    level_zero_cutoff = 3.0
    supercell_diag = (2, 2, 2)  # The triclinic cell requires replication

    cell = fixture_periodic_triclinic["cell"]
    pos = fixture_periodic_triclinic["positions"]
    chg = fixture_periodic_triclinic["charges"]
    (n_particles, n_dim) = pos.shape
    energy_ref = fixture_periodic_triclinic["energy"]
    forces_ref = fixture_periodic_triclinic["forces"]

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
        supercell_diag=supercell_diag,  # TODO: Automatic? Consistent handling in all tests?
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    # TODO: Define the error tolerances somewhere?
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=level_one_spacings,
        level_zero_cutoff=level_zero_cutoff,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
        supercell_diag=supercell_diag,  # TODO: Automatic? Consistent handling in all tests?
    )
    evaluation_fns_dyncell = create_msm(params_staticcell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(pos, chg)
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(pos, chg)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)
