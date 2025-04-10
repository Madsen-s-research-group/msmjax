from pathlib import Path

import jax
import numpy as onp
import pytest

from msmjax.benchmark_tools import path_input_structures
from msmjax.bspline_interpolation.gridops import (
    create_compute_f_oneplus_via_potential,
    create_compute_U_oneplus_direct,
    create_compute_U_oneplus_via_potential,
)
from msmjax.convenience import (
    set_up_kernels_grids_and_stencils,
    suggest_msm_params,
)

# jax.config.update("jax_enable_x64", True) # TODO


@pytest.fixture
def fixture_dir_structures() -> Path:
    """Get directory where pre-generated structures are located."""
    # TODO: Using the structures from "data/benchmark" for tests as well is not
    #  very consistent
    return path_input_structures


@pytest.fixture(params=[500, 1000, 3000])
def fixture_structure(fixture_dir_structures, request) -> dict:
    """Get one pre-generated structure with a given number of particles."""
    structsfile = fixture_dir_structures / (
        "structures_" + str(request.param) + ".npz"
    )
    structures = onp.load(structsfile)
    return {
        "cell": structures["cells"][0],
        "positions": structures["positions"][0],
        "charges": structures["charges"][0],
    }


@pytest.fixture(params=[(False, False, False), (True, True, True)])
def fixture_pbc(request) -> tuple:
    return request.param


@pytest.fixture
def fixture_base_msm_params() -> dict:
    # TODO: (something like) this would be better placed in conftest.py
    return {
        "level_one_gridspacing": 1.0,
        "alpha": 3.0,
    }


def test_calculate_different_ways(
    fixture_structure, fixture_pbc, fixture_base_msm_params
):
    """Test equality of different ways to calculate long-range energy/force"""
    box_lengths = onp.diag(fixture_structure["cell"])
    msm_params_full = suggest_msm_params(
        box_lengths=box_lengths,
        pbc=fixture_pbc,
        n_particles=fixture_structure["positions"].shape[0],
        **fixture_base_msm_params,
    )
    kernels, grids, kernel_stencils = set_up_kernels_grids_and_stencils(
        box_lengths=box_lengths, pbc=fixture_pbc, **msm_params_full
    )
    setup_params = {
        "grids": grids,
        "kernel_stencils": kernel_stencils,
    }

    compute_energy_direct = jax.jit(
        create_compute_U_oneplus_direct(**setup_params)
    )

    compute_energy_via_potential = jax.jit(
        create_compute_U_oneplus_via_potential(**setup_params)
    )
    compute_forces_via_potential = jax.jit(
        create_compute_f_oneplus_via_potential(**setup_params)
    )

    @jax.jit
    def compute_forces_grad_energy_direct(pos, chg):
        return -jax.grad(compute_energy_direct)(pos, chg)

    @jax.jit
    def compute_forces_grad_energy_via_potential(pos, chg):
        return -jax.grad(compute_energy_via_potential)(pos, chg)

    pos, chg = fixture_structure["positions"], fixture_structure["charges"]
    # pos = pos.astype(jnp.float64) # TODO
    # chg = chg.astype(jnp.float64) # TODO

    assert onp.isclose(
        compute_energy_direct(pos, chg),
        compute_energy_via_potential(pos, chg),
        atol=1.0e-6,  # TODO
    )
    assert onp.allclose(
        compute_forces_via_potential(pos, chg),
        compute_forces_grad_energy_via_potential(pos, chg),
        atol=1.0e-6,  # TODO
    )
    assert onp.allclose(
        compute_forces_via_potential(pos, chg),
        compute_forces_grad_energy_direct(pos, chg),
        atol=1.0e-6,  # TODO
    )
