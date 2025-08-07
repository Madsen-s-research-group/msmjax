import argparse
import copy
import time
import weakref
from typing import IO, Any, Union

import ase.io
import numpy as np
from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext


def md_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputstruct", type=str, required=True)
    parser.add_argument("--timestep_fs", type=float, default=5.0)
    parser.add_argument("--loginterval_fs", type=float, default=500.0)
    parser.add_argument("--simtime_ps", type=float, default=10.0)
    parser.add_argument(
        "--temp_K",
        type=float,
        default=None,
        # TODO: change help message (for NVT and NPT it is not just the initial
        #  temp, but also the thermostat target temp; or should these two options be separate?)
        help="Temperature in K according to which to set initial velocities. If not given, velocities are left at zero or values from input atoms file, if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        # TODO: Is this used only for the initial velocities, or sth else too?
        help="Random seed for setting the initial velocities.",
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
