import argparse
import copy
import time
import weakref
from typing import IO, Any, Literal, Union

import ase.io
import numpy as np
from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext


def md_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputstruct", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--timestep_fs", type=float, required=True)
    parser.add_argument("--loginterval_fs", type=float, required=True)
    parser.add_argument("--simtime_ps", type=float, required=True)
    parser.add_argument(
        "--ensemble", type=str, choices=["NVE", "NVT", "NPT"], required=True
    )
    parser.add_argument(
        "--temp_K",
        type=float,
        default=None,
        # TODO: change help message (for NVT and NPT it is not just the initial
        #  temp, but also the thermostat target temp; or should these two options be separate?)
        help="Temperature in K according to which to set initial velocities. If not given, velocities are left at zero or values from input atoms file, if available.",
    )
    parser.add_argument(
        "--pressure_GPa", type=float, default=None, help="Pressure in GPa"
    )

    return parser


class CustomMDLogger(IOContext):
    """Class for logging molecular dynamics simulations.

    Adapted from ase.md.MDLogger

    Parameters:
    dyn:           The dynamics.  Only a weak reference is kept.

    atoms:         The atoms.

    logfile:       File name or open file, "-" meaning standard output.

    stress=False:  Include stress in log.

    peratom=False: Write energies per atom.

    mode="a":      How the file is opened if logfile is a filename.
    """

    def __init__(
        self,
        dyn: Any,  # not fully annotated so far to avoid a circular import
        atoms: Atoms,
        logfile: Union[IO, str],
        header: bool = True,
        stress: bool = False,
        peratom: bool = False,
        mode: str = "a",
        comm=world,
    ):
        self.dyn = weakref.proxy(dyn) if hasattr(dyn, "get_time") else None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)
        self.stress = stress
        self.peratom = peratom
        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s  %6s" % (
                "Etot/N[eV]",
                "Epot/N[eV]",
                "Ekin/N[eV]",
                "T[K]",
            )
            self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            self.hdr += "%19s %10s %12s %12s %12s  %6s" % (
                "Walltime[s]",
                "Step",
                "Etot[eV]",
                "Epot[eV]",
                "Ekin[eV]",
                "T[K]",
            )
            # Choose a sensible number of decimals
            # TODO: More decimals (Energies don't seem to be saved to the trajectory
            #  when using a MixedCalculator, so the log file is currently the only place
            #  where the energies are saved. So might as well save them accurately.)
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += (
                "%.7f" + "%11i " + 3 * ("%%12.%df " % (digits,)) + " %6.1f"
            )
        if self.stress:
            self.hdr += (
                "      ---------------------- stress [GPa] "
                "-----------------------"
            )
            self.fmt += 6 * " %10.3f"
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        global_natoms = self.atoms.get_global_number_of_atoms()
        t_wall = time.time()
        n_steps = self.dyn.nsteps
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (t_wall, n_steps, epot + ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(
                self.atoms.get_stress(include_ideal_gas=True) / units.GPa
            )
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()


def read_logfile(logfile, stress=False, max_rows=None):
    loaded = np.atleast_2d(np.loadtxt(logfile, skiprows=1, max_rows=max_rows))
    results = {
        "time_ps": loaded[:, 0],
        "wall_time": loaded[:, 1],
        "step": loaded[:, 2].astype(int),
        "e_tot": loaded[:, 3],
        "e_pot": loaded[:, 4],
        "e_kin": loaded[:, 5],
        "temp_K": loaded[:, 6],
    }

    if stress:
        results["stress_GPa"] = loaded[:, 7:]

    return results


def fold_trajectory(traj):
    traj_folded = copy.deepcopy(traj)
    for at in traj_folded:
        at.wrap()
    return traj_folded


def read_traj_and_log(trajfile, logfile):
    traj = ase.io.read(trajfile, index=":")

    for i in range(len(traj)):
        last_not_nan_idx = i
        if (
            np.isnan(traj[i].calc.results["energy"])
            or np.isnan(traj[i].cell[...]).any()
        ):
            last_not_nan_idx -= 1
            break

    traj = traj[: last_not_nan_idx + 1]
    traj = fold_trajectory(traj)
    log = read_logfile(logfile, stress=True, max_rows=last_not_nan_idx + 1)

    return traj, log


def make_convert_lj_to_ase(
    ref_epsilon: float,
    ref_sigma: float,
    n_dim: int = 3,
):
    """Create function that converts from Lennard-Jones units to ASE units.

    The conversion of times should probably not be trusted.

    Args:
        ref_epsilon: Depth of Lennard-Jones potential, in ASE's units. Serves as
            reference value for the energy.
        ref_sigma: Distance at which the Lennard-Jones potential crosses zero,
            in ASE's units. Serves as reference value for lengths.
        ref_mass: Reference mass.
        n_dim: Spatial dimension.
    """

    def convert(
        energy: float = None,
        length: float = None,
        density: float = None,
        pressure: float = None,
        temperature: float = None,
    ):
        out = {}

        if energy is not None:
            out["energy"] = energy * ref_epsilon
        if length is not None:
            out["length"] = length * ref_sigma
        if density is not None:
            out["density"] = density / ref_sigma**n_dim
        if pressure is not None:
            out["pressure"] = pressure * ref_epsilon / ref_sigma**n_dim
        if temperature is not None:
            out["temperature"] = temperature * ref_epsilon / ase.units.kB

        return out

    return convert


def make_md_command(
    simtime_ps: float,
    timestep_fs: float,
    loginterval_fs: float,
    ensemble: Literal["NVE", "NVT", "NPT"],
    filename_md_script: str,
    filename_input_struct: str,
    filename_msm_params: str,
    dirname_out: str,
    temp_K: float | None = None,
    pressure_GPa: float | None = None,
):

    command = rf"""python {filename_md_script} \
    --inputstruct {filename_input_struct} \
    --msm_params {filename_msm_params} \
    --outdir {dirname_out} \
    --timestep_fs {timestep_fs} \
    --loginterval_fs {loginterval_fs} \
    --simtime_ps {simtime_ps} \
    --ensemble {ensemble}"""

    if ensemble in ["NVT", "NPT"]:
        if temp_K is None:
            raise ValueError("temperature required in NVT and NPT ensembles")
        command += " \\\n"
        command += f"    --temp_K {temp_K}"
        if ensemble == "NPT":
            if pressure_GPa is None:
                raise ValueError("pressure required in NPT ensemble")
            command += " \\\n"
            command += f"    --pressure_GPa {pressure_GPa}"

    command += "\n\n"

    return command


def read_logfile_ase(logfile, stress=False):
    loaded = np.loadtxt(logfile, skiprows=1)
    results = {
        "time_ps": loaded[:, 0],
        "wall_time": loaded[:, 1],
        "step": loaded[:, 2].astype(int),
        "Etot_eV": loaded[:, 3],
        "Epot_eV": loaded[:, 4],
        "Ekin_eV": loaded[:, 5],
        "temp_K": loaded[:, 6],
    }
    if stress:
        results["stress_GPa"] = loaded[:, 7:]

    return results


def fold_trajectory_ase(traj):
    traj_folded = copy.deepcopy(traj)
    for at in traj_folded:
        at.wrap()
    return traj_folded


def calculate_moving_average(arr, window_size, pad_with_nan=True):
    if len(arr) < 2 * window_size // 2:
        return np.full_like(arr, np.nan)

    moving_average = np.convolve(
        arr,
        np.ones(window_size) / window_size,
        mode="valid",
    )
    if pad_with_nan:
        return np.pad(
            moving_average,
            pad_width=(window_size // 2, window_size // 2),
            mode="constant",
            constant_values=np.nan,
        )
    return moving_average
