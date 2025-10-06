"""Integration tests: End-to-end pipeline for evaluating various quantities"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import jax
import numpy as onp
import pytest

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import (
    calc_relative_rmse_percent,
)

# For closeness checks to pass in single precision
ATOL = 5.0e-6
LEVEL_ONE_SPACINGS = 1.0
LEVEL_ZERO_CUTOFF = 4.0
P = 6


@pytest.fixture(scope="module")
def fixture_datadir() -> Path:
    """Path to directory containing structures and reference results"""
    return Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="module")
def fixture_nonperiodic_cubic(fixture_datadir):
    """Data and settings characterizing nonperiodic cubic structure"""
    data = onp.load(fixture_datadir / "nonperiodic_cubic.npz")
    pbc = (False,) * 3
    cell_mode = "ortho"
    supercell_diag = None
    return data, pbc, cell_mode, supercell_diag


@pytest.fixture(scope="module")
def fixture_nonperiodic_ortho_diff_sides(fixture_datadir):
    """Data and settings characterizing nonperiodic orthorhombic structure"""
    data = onp.load(
        fixture_datadir / "nonperiodic_ortho-different-sidelengths.npz"
    )
    pbc = (False,) * 3
    cell_mode = "ortho"
    supercell_diag = None
    return data, pbc, cell_mode, supercell_diag


@pytest.fixture(scope="module")
def fixture_nonperiodic_triclinic(fixture_datadir):
    """Data and settings characterizing nonperiodic triclinic structure"""
    data = onp.load(fixture_datadir / "nonperiodic_triclinic.npz")
    pbc = (False,) * 3
    cell_mode = "triclinic"
    supercell_diag = None
    return data, pbc, cell_mode, supercell_diag


@pytest.fixture(scope="module")
def fixture_periodic_cubic(fixture_datadir):
    """Data and settings characterizing periodic cubic structure"""
    data = onp.load(fixture_datadir / "periodic_cubic.npz")
    pbc = (True,) * 3
    cell_mode = "ortho"
    supercell_diag = (2, 2, 2)
    return data, pbc, cell_mode, supercell_diag


@pytest.fixture(scope="module")
def fixture_periodic_ortho_diff_sides(fixture_datadir):
    """Data and settings characterizing periodic orthorhombic structure"""
    data = onp.load(
        fixture_datadir / "periodic_ortho-different-sidelengths.npz"
    )
    pbc = (True,) * 3
    cell_mode = "ortho"
    supercell_diag = (1, 1, 2)
    return data, pbc, cell_mode, supercell_diag


@pytest.fixture(scope="module")
def fixture_periodic_triclinic(fixture_datadir):
    """Data and settings characterizing periodic triclinic structure"""
    data = onp.load(fixture_datadir / "periodic_triclinic.npz")
    pbc = (True,) * 3
    cell_mode = "triclinic"
    supercell_diag = (2, 2, 2)
    return data, pbc, cell_mode, supercell_diag


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
    """Test that MSM and reference results agree for various quantities"""
    data, pbc, cell_mode, supercell_diag = fixture_system_definition
    pos = data["positions"]
    chg = data["charges"]
    cell = data["cell"]
    energy_ref = data["energy"]
    forces_ref = data["forces"]
    chargegrad_ref = data["charge_gradient"]
    stress_ref = data["stress"]

    (n_particles, n_dim) = pos.shape

    params_staticcell = set_up_msm_params(
        cell=cell,
        level_one_spacings=LEVEL_ONE_SPACINGS,
        level_zero_cutoff=LEVEL_ZERO_CUTOFF,
        p=P,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=False,
        n_particles=n_particles,
        supercell_diag=supercell_diag,
    )
    evaluation_fns_staticcell = create_msm(params_staticcell)
    energy_msm_staticcell = jax.jit(evaluation_fns_staticcell["energy"])(
        pos, chg
    )
    forces_msm_staticcell = jax.jit(evaluation_fns_staticcell["forces"])(
        pos, chg
    )
    chargegrad_msm_staticcell = jax.jit(
        evaluation_fns_staticcell["charge_gradient"]
    )(pos, chg)
    assert onp.abs(energy_msm_staticcell - energy_ref) < 0.25
    assert calc_relative_rmse_percent(forces_msm_staticcell, forces_ref) < 1.0
    assert (
        calc_relative_rmse_percent(chargegrad_msm_staticcell, chargegrad_ref)
        < 1.0
    )

    params_dyncell = set_up_msm_params(
        cell=cell,
        level_one_spacings=LEVEL_ONE_SPACINGS,
        level_zero_cutoff=LEVEL_ZERO_CUTOFF,
        p=P,
        pbc=pbc,
        cell_mode=cell_mode,
        dynamic_cell=True,
        n_particles=n_particles,
        supercell_diag=supercell_diag,
    )
    evaluation_fns_dyncell = create_msm(params_dyncell)
    energy_msm_dyncell = jax.jit(evaluation_fns_dyncell["energy"])(
        pos, chg, cell
    )
    forces_msm_dyncell = jax.jit(evaluation_fns_dyncell["forces"])(
        pos, chg, cell
    )
    chargegrad_msm_dyncell = jax.jit(
        evaluation_fns_dyncell["charge_gradient"]
    )(pos, chg, cell)
    assert onp.isclose(energy_msm_dyncell, energy_msm_staticcell, atol=ATOL)
    assert onp.allclose(forces_msm_dyncell, forces_msm_staticcell, atol=ATOL)
    assert onp.allclose(
        chargegrad_msm_dyncell, chargegrad_msm_staticcell, atol=ATOL
    )

    stress_msm = jax.jit(evaluation_fns_dyncell["stress"])(pos, chg, cell)
    if cell_mode == "ortho":
        # In cell_mode "ortho", only the diagonal components of the stress
        # tensor are calculated correctly and returned.
        assert calc_relative_rmse_percent(stress_msm, stress_ref[:3]) < 5.0
    else:
        assert calc_relative_rmse_percent(stress_msm, stress_ref) < 5.0
