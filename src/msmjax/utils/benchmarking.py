import contextlib
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import ase.io
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import numpy as onp
import numpy.typing as npt
from ase import Atoms

path_input_structures = (
    Path(__file__).resolve().parents[3] / "data" / "benchmark" / "structures"
)
path_reference_lammps_p3m = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "benchmark"
    / "results_ref_periodic_lammps_p3m"
)

# TODO: clean up (not all of these functions need to be published)


def get_git_commit_id(repository_path):
    proc = subprocess.run(
        shlex.split(f"git -C {Path(repository_path).resolve()} show -s"),
        capture_output=True,
    )
    commit_id = proc.stdout.decode("utf8").split()[1]
    return commit_id


def get_git_branch(repository_path):
    proc = subprocess.run(
        shlex.split(f"git -C {Path(repository_path).resolve()} status"),
        capture_output=True,
    )
    branch_name = proc.stdout.decode("utf8").split()[2]
    return branch_name


def get_repository_info(path):
    try:
        commit_id = get_git_commit_id(path)
        branch_name = get_git_branch(path)
        return {"path": path, "branch": branch_name, "commit_id": commit_id}
    except Exception as e:
        return {
            "path": path,
            "branch": None,
            "commit_id": None,
            "exception": repr(e),
        }


def get_metadata(additional_repository_paths: dict = None):
    metadata = {
        "timestamp": str(datetime.now()),
        "system_info": {
            "hostname": socket.gethostname(),
            "uname": platform.uname()._asdict(),
        },
        "python_info": {"version": sys.version, "path": sys.path},
        "jax_info": {
            "jax.__version__": jax.__version__,
            "jaxlib.__version__": jaxlib.__version__,
            "jax_device": jax.devices()[0].device_kind,
        },
    }

    path_msmjax = str(Path(__file__).resolve().parents[2])
    repository_info_msmjax = get_repository_info(path_msmjax)
    metadata["repository_info_msmjax"] = repository_info_msmjax

    if additional_repository_paths is not None:
        additional_repository_info = {}
        for label, path in additional_repository_paths.items():
            additional_repository_info[label] = get_repository_info(path)
        metadata["additional_repository_info"] = additional_repository_info

    return metadata


def time_set_of_structures(structures, pbc, setup_fn, **setup_fn_kwargs):
    timed_calc, info = setup_fn(
        structures=structures,
        pbc=pbc,
        **setup_fn_kwargs,
    )
    times_all = []
    for idx_structure in range(len(structures["positions"])):
        pos = structures["positions"][idx_structure]
        chg = structures["charges"][idx_structure]
        cell = structures["cells"][idx_structure]
        pos = jax.device_put(pos)
        chg = jax.device_put(chg)
        cell = jax.device_put(cell)
        times_all.append(timed_calc(pos, chg, cell))

    output = {
        "times": onp.array(times_all).tolist(),
        "info": info,
    }

    return output


@contextlib.contextmanager
def dir_context(dir_name: Path):
    """Create a context to run code in a different directory.

    Args:
        dir_name: Route to the directory.
    """
    cwd = os.getcwd()
    try:
        os.chdir(dir_name)
        yield
    finally:
        os.chdir(cwd)


def write_lammps_data(filename, cell, positions, charges) -> None:
    """Write structure as LAMMPS data file"""
    n_particles = positions.shape[0]
    n_dims = positions.shape[1]
    atoms = Atoms(
        symbols=["X"] * n_particles,
        cell=cell,
        positions=positions,
        charges=charges,
        pbc=[True] * n_dims,
    )
    ase.io.write(filename, atoms, format="lammps-data", atom_style="charge")


def parse_energy_from_lammps_log(filename) -> float:
    """Get the energy from LAMMPS log file"""
    # TODO: unify with the other log parser function?
    with open(filename, "r") as f:
        for line in f:
            if "PotEng" in line:
                line_poteng = f.readline()
                energy = float(line_poteng.strip())
                return energy

    raise ValueError("EOF reached without finding energy")


def parse_lammps_log(filename) -> tuple[float, onp.ndarray]:
    """Get energy and other results from LAMMPS log file"""
    # TODO: unify with the other log parser function?
    with open(filename, "r") as f:
        for line in f:
            if "PotEng" in line:
                line_header = line
                line_values = f.readline()
                resultsdict = {
                    k: float(v)
                    for k, v in zip(line_header.split(), line_values.split())
                }
                energy = resultsdict["PotEng"]
                # TODO: minus or not?
                stress = -onp.array(
                    [
                        resultsdict[k]
                        for k in ["Pxx", "Pyy", "Pzz", "Pxy", "Pxz", "Pyz"]
                    ]
                )
                return energy, stress

    raise ValueError(
        "Error parsing LAMMPS log file: EOF reached without finding thermo "
        "output line"
    )


# We want to calculate the value of (q_i * q_j) / r_{ij}, in whatever units
# q_i, q_j, r_i, r_j are supplied in, and whatever quantities they might actually
# represent.
# LAMMPS calculates energy = (1 / (4 * pi * epsilon_0)) * ((q_i * q_j) / r_{ij}),
# with (for style `units metal`) q_i, q_j in units of elementary charge,
# r_i, r_j in Angstrom, and energy in eV. Plugging in the values for epsilon_0,
# Angstrom, eV, Coulomb, in SI units, the conversion factor is:
CONVERSION_FACTOR = (4 * onp.pi) * 8.8541878128 / 1.602176634 / 10**3

# TODO: Explain where this value comes from
CONVERSION_FACTOR_STRESS = 4.334_488_014_869e-08


def make_lammps_input_text_pppm(
    filename_data,
    filename_dump,
    accuracy: float,
    max_neighbors_one_atom: int | None,
):
    """Write LAMMPS input script"""
    if max_neighbors_one_atom is None:
        neigh_line = ""
    else:
        neigh_line = f"neigh_modify one {max_neighbors_one_atom}"

    text = f"""# 1) Initialization
units metal
dimension 3
boundary p p p
atom_style charge
pair_style coul/long 10.0
kspace_style pppm {accuracy:.10g}
{neigh_line}

# 2) System definition
read_data {filename_data}
kspace_style pppm {accuracy:.10g}  # need to reinitialize after reading data to work for triclinic cells

# 3) Simulation settings
mass 1 1
pair_coeff * *

# 4) Output settings
compute 1 all pressure NULL virial
compute peratom all pe/atom
thermo 1
thermo_style custom pe pxx pyy pzz pxy pxz pyz
dump mydmp all custom 1 {filename_dump} id type x y z fx fy fz c_peratom

# 5) Run
run 0
"""
    return text


def eval_lammps_pppm(
    positions: npt.ArrayLike,
    charges: npt.ArrayLike,
    cell: npt.ArrayLike,
    lammps_executable: str = "lmp",
    accuracy: float = 1.0e-5,
    max_neighbors_one_atom: int | None = None,
    show_stdout=False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper to compute periodic electrostatic energy, forces in LAMMPS with p3m

    Args:
        positions: Array of article positions, shape `(n_particles, 3)`
        charges: Array of particle charges, `(n_particles,)`
        cell: Unit cell
        lammps_executable: Path to LAMMPS executable
        max_neighbors_one_atom: Maximum number of neighbors of a single atom.
            Supplied to LAMMPS via `neigh_modify one` if given. You may need
            to increase this value if you're getting errors.

    Returns:
        energy, forces
    """
    # TODO: Change to the newer version of this function, with interface
    #  similar to eval_lammps_msm
    filename_lammps_data = "structure.data"
    filename_lammps_dump = "dump.lammpstrj"
    filename_lammps_log = "log.lammps"
    filename_lammps_in = "input.lammps"

    with tempfile.TemporaryDirectory() as folder_name:
        with dir_context(folder_name):
            write_lammps_data(
                filename=filename_lammps_data,
                cell=cell,
                positions=positions,
                charges=charges,
            )
            lammps_input_text = make_lammps_input_text_pppm(
                filename_data=filename_lammps_data,
                filename_dump=filename_lammps_dump,
                accuracy=accuracy,
                max_neighbors_one_atom=max_neighbors_one_atom,
            )
            with open(filename_lammps_in, "w") as f:
                f.write(lammps_input_text)

            subprocess_args = [lammps_executable, "-in", filename_lammps_in]
            if show_stdout:
                subprocess.run(subprocess_args)
            else:
                subprocess.run(subprocess_args, stdout=subprocess.DEVNULL)

            energy, stress = parse_lammps_log(filename_lammps_log)
            atoms_loaded_dump = ase.io.read(filename_lammps_dump)
            forces = atoms_loaded_dump.calc.results["forces"]
            # TODO: Make sure the formula for this is actually correct
            #  (especially considering interactions of atoms with their own
            #  images under pbc?)
            energy_peratom = atoms_loaded_dump.arrays["c_peratom"].squeeze()
            charge_gradient = 2 * energy_peratom / charges

    # TODO: stress unit conversion
    return (
        CONVERSION_FACTOR * energy,
        CONVERSION_FACTOR * forces,
        CONVERSION_FACTOR * charge_gradient,
        CONVERSION_FACTOR_STRESS * stress,
    )


def make_lammps_input_text_msm(
    filename_data,
    filename_dump,
    pbc: Sequence[bool],
    cutoff: float,
    accuracy: float,
    max_neighbors_one_atom: int | None,
):
    """Write LAMMPS input script"""
    if max_neighbors_one_atom is None:
        neigh_line = ""
    else:
        neigh_line = f"neigh_modify one {int(max_neighbors_one_atom)}"

    pbcstring = " ".join(["p" if periodic else "f" for periodic in pbc])

    text = f"""# 1) Initialization
units metal
dimension 3
boundary {pbcstring}
atom_style charge
pair_style coul/msm {cutoff:f}
pair_modify table 0
kspace_style msm {accuracy:.10g}
{neigh_line}
neigh_modify page 6000000    # TODO: make dependent on neigh_line

# 2) System definition
read_data {filename_data}
kspace_style msm {accuracy:.10g}  # need to reinitialize after reading data to work for triclinic cells

# 3) Simulation settings
mass 1 1
pair_coeff * *

# 4) Output settings
thermo 1
thermo_style custom pe
dump mydmp all custom 1 {filename_dump} id type x y z fx fy fz

# 5) Run
run 0
"""
    return text


def eval_lammps_msm(
    positions: npt.ArrayLike,
    charges: npt.ArrayLike,
    cell: npt.ArrayLike,
    pbc: Sequence[bool],
    lammps_executable: str = "lmp",
    cutoff: float = 10.0,
    accuracy: float = 1.0e-5,
    max_neighbors_one_atom: int | None = None,
    show_stdout=False,
) -> Tuple[float, np.ndarray]:
    """Wrapper to compute periodic electrostatic energy, forces in LAMMPS with p3m

    Args:
        positions: Array of article positions, shape `(n_particles, 3)`
        charges: Array of particle charges, `(n_particles,)`
        cell: Unit cell
        lammps_executable: Path to LAMMPS executable
        max_neighbors_one_atom: Maximum number of neighbors of a single atom.
            Supplied to LAMMPS via `neigh_modify one` if given. You may need
            to increase this value if you're getting errors.

    Returns:
        energy, forces
    """
    filename_lammps_data = "structure.data"
    filename_lammps_dump = "dump.lammpstrj"
    filename_lammps_log = "log.lammps"
    filename_lammps_in = "input.lammps"

    with tempfile.TemporaryDirectory() as folder_name:
        with dir_context(folder_name):
            write_lammps_data(
                filename=filename_lammps_data,
                cell=cell,
                positions=positions,
                charges=charges,
            )
            lammps_input_text = make_lammps_input_text_msm(
                filename_data=filename_lammps_data,
                filename_dump=filename_lammps_dump,
                pbc=pbc,
                cutoff=cutoff,
                accuracy=accuracy,
                max_neighbors_one_atom=max_neighbors_one_atom,
            )
            with open(filename_lammps_in, "w") as f:
                f.write(lammps_input_text)

            subprocess_args = [lammps_executable, "-in", filename_lammps_in]
            if show_stdout:
                subprocess.run(subprocess_args)
            else:
                subprocess.run(subprocess_args, stdout=subprocess.DEVNULL)

            energy = parse_energy_from_lammps_log(filename_lammps_log)
            forces = ase.io.read(filename_lammps_dump).calc.results["forces"]

    return CONVERSION_FACTOR * energy, CONVERSION_FACTOR * forces


def calc_rmse(y_pred, y_true):
    return onp.sqrt(((y_pred - y_true) ** 2).mean())


def calc_relative_rmse_percent(y_pred, y_true):
    return calc_rmse(y_pred, y_true) / y_true.std() * 100


def plot_parity_line(ax, **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    common_lims = (np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]]))
    p = ax.plot(common_lims, common_lims, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return p


if __name__ == "__main__":
    LAMMPS_EXECUTABLE = "/home/florian/Downloads/lammps-static/bin/lmp"

    structures = onp.load(path_input_structures / "structures_500.npz")
    idx = 0
    pos = structures["positions"][idx]
    chg = structures["charges"][idx]
    cell = structures["cells"][idx]

    energy, forces = evaluate_structure_with_lammps_p3m(
        positions=pos,
        charges=chg,
        cell=cell,
        lammps_executable=LAMMPS_EXECUTABLE,
        max_neighbors_one_atom=10000,
    )

    print()
    print(f"energy: {energy}")
    print(f"forces: {forces}")
