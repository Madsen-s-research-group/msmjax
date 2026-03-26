import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

import sys
from functools import partial
from pathlib import Path

import ase.io
import ase.units
import jax.numpy as jnp
import numpy as onp
from ase.md import VelocityVerlet
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT

import msmjax
from msmjax.calculators import MSMParams

parentpath_charged_lj_utils = str(
    Path(msmjax.__file__).resolve().parents[2] / "examples" / "charged_lj_md"
)
sys.path.append(parentpath_charged_lj_utils)
from utils_charged_lj.energy_model import (
    EPSILON_ELECTRONVOLT,
    SIGMA_ANGSTROM,
    GenericWrapperCalculator,
    make_charged_lj_evaluation_fns,
)
from utils_charged_lj.helpers import CustomMDLogger, md_base_parser

if __name__ == "__main__":
    ################################################################################
    # Parsing, read input structure
    ################################################################################
    parser = md_base_parser()
    parser.add_argument(
        "--msm_params",
        type=str,
        help="Path to json file containing MSM parameters",
    )
    args = parser.parse_args()

    atoms = ase.io.read(args.inputstruct)
    n_particles = len(atoms)
    n_dim = atoms.get_positions().shape[1]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True)

    ################################################################################
    # Energy model and calculator setup
    ################################################################################
    msm_params = MSMParams.load_json(args.msm_params)
    evaluation_fns = make_charged_lj_evaluation_fns(
        msm_params=msm_params,
        sigma=SIGMA_ANGSTROM,
        epsilon=EPSILON_ELECTRONVOLT,
    )

    effective_charges = atoms.get_initial_charges()

    energy_fn = partial(evaluation_fns["energy"], charges=effective_charges)
    forces_fn = partial(evaluation_fns["forces"], charges=effective_charges)

    # If msmJAX was set up with cell_mode = "ortho", the return value for the
    # stress consists only of the three diagonal elements, and we need to pad
    # it to six elements:

    def stress_fn_6_comp(positions, charges, cell):
        stresses_diag = evaluation_fns["stress"](positions, charges, cell)
        return jnp.pad(stresses_diag, pad_width=(0, 3), constant_values=0.0)

    atoms.calc = GenericWrapperCalculator(
        energy_fn=partial(evaluation_fns["energy"], charges=effective_charges),
        forces_fn=partial(evaluation_fns["forces"], charges=effective_charges),
        stress_fn=partial(stress_fn_6_comp, charges=effective_charges),
    )

    ################################################################################
    # Integrator setup
    ################################################################################
    n_steps = int(1000 * args.simtime_ps / args.timestep_fs)
    write_every_ith_step = max(1, int(args.loginterval_fs / args.timestep_fs))
    timestep = args.timestep_fs * ase.units.fs
    temp_K = args.temp_K

    trajfile = outdir / "md.traj"
    logfile = outdir / "md.log"

    common_dynamics_kwargs = dict(
        timestep=timestep,
        trajectory=ase.io.Trajectory(
            trajfile, atoms=atoms, mode="w", properties=["energy", "stress"]
        ),
        loginterval=write_every_ith_step,
    )

    if args.ensemble == "NVE":
        dyn = VelocityVerlet(atoms, **common_dynamics_kwargs)
    elif args.ensemble == "NVT":
        dyn = NoseHooverChainNVT(
            atoms,
            temperature_K=temp_K,
            tdamp=100 * timestep,
            **common_dynamics_kwargs,
        )
    elif args.ensemble == "NPT":
        pressure_au = args.pressure_GPa / 1.602176634 / 10**2
        dyn = IsotropicMTKNPT(
            atoms,
            temperature_K=temp_K,
            pressure_au=pressure_au,
            tdamp=100 * timestep,
            pdamp=1000 * timestep,
            **common_dynamics_kwargs,
        )

    dyn.attach(
        CustomMDLogger(dyn, atoms, logfile, stress=True),
        interval=write_every_ith_step,
    )
    ################################################################################
    # Run MD
    ################################################################################
    print(
        f"- Running MD for {n_steps} steps (check {logfile} to monitor progress)"
    )
    dyn.run(n_steps)
