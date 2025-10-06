"""Utilities for benchmarking"""

import contextlib
import os
import subprocess
import tempfile
import timeit
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import ase.io
import jax
import jax.numpy as jnp
import matscipy.neighbours
import numpy as np
import numpy as onp
import numpy.typing as npt
from ase import Atoms
from jax import Array
from jax.typing import ArrayLike
from tqdm import tqdm

from msmjax.core.shortrange import make_eval_pair_pot
from msmjax.utils.general import inds_matrix_to_six_component_stress

path_input_structures = (
    Path(__file__).resolve().parents[3] / "data" / "benchmark" / "structures"
)


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
    with open(filename, "r") as f:
        for line in f:
            if "PotEng" in line:
                line_poteng = f.readline()
                energy = float(line_poteng.strip())
                return energy

    raise ValueError("EOF reached without finding energy")


def parse_lammps_log(filename) -> tuple[float, onp.ndarray]:
    """Get energy and other results from LAMMPS log file"""
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

# The stress conversion factor follows from analytically expressing the
# electrostatic pressure for some simple crystal structure via its
# Madelung constant:
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
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper to compute periodic electrostatic energy, forces in LAMMPS with p3m

    Args:
        positions: Array of article positions, shape `(n_particles, 3)`
        charges: Array of particle charges, `(n_particles,)`
        cell: Unit cell
        lammps_executable: Path to LAMMPS executable
        accuracy: Accuracy setting passed to LAMMPS via
            `kspace_style pppm {accuracy}`.
        max_neighbors_one_atom: Maximum number of neighbors of a single atom.
            Supplied to LAMMPS via `neigh_modify one` if given. You may need
            to increase this value if you're getting errors.
        show_stdout: Whether to show the stdout from the subprocess call to
            LAMMPS.

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
            energy_peratom = atoms_loaded_dump.arrays["c_peratom"].squeeze()
            charge_gradient = 2 * energy_peratom / charges

    return (
        CONVERSION_FACTOR * energy,
        CONVERSION_FACTOR * forces,
        CONVERSION_FACTOR * charge_gradient,
        CONVERSION_FACTOR_STRESS * stress,
    )


def calc_rmse(y_pred: ArrayLike, y_true: ArrayLike) -> Array:
    """Calculate RMSE."""
    return onp.sqrt(((y_pred - y_true) ** 2).mean())


def calc_relative_rmse_percent(y_pred: ArrayLike, y_true: ArrayLike) -> Array:
    """Calculate RMSE normalized to STD of reference results, as percentage."""
    return calc_rmse(y_pred, y_true) / y_true.std() * 100


def calc_relative_rmse(y_pred: ArrayLike, y_true: ArrayLike) -> Array:
    """Calculate RMSE normalized to STD of reference results."""
    return calc_rmse(y_pred, y_true) / y_true.std()


def make_timed_eval(
    fn: Callable, repeat: int = 10, number: int = 100
) -> Callable:
    """Given a JAX-jittable function, return a function to time its evaluation.

    The returned function internally takes care of jitting the function to be
    timed, and of calling ``.block_until_ready()`` on its output to avoid
    meaningless results because of JAX's asynchronous dispatch.

    Args:
        fn: A JAX-jittable function.
        repeat: Passed to ``timeit.repeat``.
        number: Passed to ``timeit.repeat``.

    Returns:
        A function with the same argument structure as ``fn`` that, when
        called, returns a 2-element tuple containing:

            - The measured evaluation time
            - The original output of ``fn``.
    """
    jitted_fn = jax.jit(fn)

    def time_model_eval(*args, **kwargs) -> tuple[float, Any]:
        # If the output is a container type, we cannot call
        # block_until_ready() on it directly, but first need to flatten
        # it down to one of the leaf arrays.
        fn_to_time = lambda: jax.tree.flatten(jitted_fn(*args, **kwargs))[0][
            0
        ].block_until_ready()

        # Call once to ensure jit-compilation
        output = fn_to_time()

        times_per_loop = timeit.repeat(
            fn_to_time, repeat=repeat, number=number
        )
        mean_times_per_call = onp.array(times_per_loop) / number

        return min(mean_times_per_call), output

    return time_model_eval


@partial(jax.jit, static_argnums=2)
def remove_duplicates_from_neighborlist(
    neighborlist: Sequence[ArrayLike], fill_value: int, size: int
) -> tuple[Array, Array]:
    """Remove duplicate index pairs from a neighbor list.

    Args:
        neighborlist: Original neighbor list that may contain duplicates.
            (like containing both index pair `ij` and `ji`).
            Tuple of two 1-d integer arrays of the same size. See
            :py:func:`msmjax.utils.benchmarking` for more details on format.
        fill_value: A value used to pad the arrays in the neighbor list
            to length ``size``. Should be a value that is ignored by the
            evaluation function to which the neighbor list will be passed.
        size: The number of index pairs (including placeholder values
            that contain ``fill_value``) in the returned duplicate-free
            neighbor list. Must be known in advance to satisfy static-shape
            constraints under JIT.

    Returns:
        The neighbor list with duplicate index pairs removed (i.e., containing
        only `ij` and not `ji`).
    """
    without_duplicates = jnp.unique(
        jnp.sort(jnp.column_stack([neighborlist[0], neighborlist[1]]), axis=1),
        axis=0,
        size=size,
        fill_value=fill_value,
    )
    return (without_duplicates[:, 0], without_duplicates[:, 1])


def build_duplicate_free_neighborlists(
    set_of_positions: Sequence[ArrayLike],
    set_of_cells: Sequence[ArrayLike],
    cutoff: float,
    pbc: Sequence[bool],
) -> list[tuple[Array, Array]]:
    """Construct a neighbor list with no duplicate index pairs.

    First uses matscipy for constructing the initial neighbor list, which does
    contain duplicates, then JAX is ued to remove the duplicates.
    The function processes several structures and pads the neighbor lists to
    the maximum neighbor-list size out of all these structures, in order to
    gain efficiency by avoiding re-compilation of the JAX duplicate-removal
    step.

    Args:
        set_of_positions: Sequence of particle positions (arrays of shape
            `(n_particles, n_dim)`), each corresponding to one structure.
        set_of_cells: Sequence of cells (arrays of shape `(n_dim, n_dim)`),
            each corresponding to one structure.
        cutoff: Cutoff radius for neighbor search.
        pbc: One boolean per direction signaling periodicity.

    Returns:
        List of duplicate-free neighbor lists for all structures.
    """
    n_structures = len(set_of_positions)

    neighborlists_raw = []
    print(f"- Building neighbor list(s) for {n_structures} structure(s)")
    for pos, cll in tqdm(
        zip(set_of_positions, set_of_cells), total=n_structures
    ):
        nbl = matscipy.neighbours.neighbour_list(
            "ij", cutoff=cutoff, positions=pos, cell=cll, pbc=pbc
        )
        neighborlists_raw.append(nbl)

    # Pad to common max length
    max_size = max([len(nbl[0]) for nbl in neighborlists_raw])
    placeholder_index = max(pos.shape[0] for pos in set_of_positions)
    for idx_structure in range(n_structures):
        i, j = neighborlists_raw[idx_structure]
        padding = max_size - len(i)
        neighborlists_raw[idx_structure] = (
            jnp.pad(i, (0, padding), constant_values=placeholder_index),
            jnp.pad(j, (0, padding), constant_values=placeholder_index),
        )

    max_size_nodupes = max_size // 2
    neighborlists_nodupes = [
        remove_duplicates_from_neighborlist(
            nbl, fill_value=placeholder_index, size=max_size_nodupes
        )
        for nbl in neighborlists_raw
    ]
    print("- Done building neighbor list(s)")

    return neighborlists_nodupes


def coulomb_kernel(r):
    return 1.0 / r


def calc_energy_nonperiodic_direct_summation(
    positions: ArrayLike, charges: ArrayLike
) -> Array:
    """Calculate exact energy for no periodicity by all-pairs sum

    Args:
        positions: Array of particle positions, shape `(n_particles, n_dim)`
        charges: Array of particle charges, shape `(n_particles,)`

    Returns:
        Energy
    """
    n_dim = positions.shape[1]
    compute_pair_term = make_eval_pair_pot(
        kernel_fn=coulomb_kernel, pbc=(False,) * n_dim
    )
    return compute_pair_term(positions, charges)


def calc_forces_nonperiodic_direct_summation(
    positions: ArrayLike, charges: ArrayLike
) -> Array:
    """Calculate exact forces for no periodicity by all-pairs sum

    Args:
        positions: Array of particle positions, shape `(n_particles, n_dim)`
        charges: Array of particle charges, shape `(n_particles,)`

    Returns:
        Forces
    """
    return -jax.grad(calc_energy_nonperiodic_direct_summation, argnums=0)(
        positions, charges
    )


def calc_chargegrad_nonperiodic_direct_summation(
    positions: ArrayLike, charges: ArrayLike
) -> Array:
    """Calculate exact charge gradient for no periodicity by all-pairs sum

    Args:
        positions: Array of particle positions, shape `(n_particles, n_dim)`
        charges: Array of particle charges, shape `(n_particles,)`

    Returns:
        Gradient of the energy w.r.t. particle charges
    """
    n_dim = positions.shape[1]
    compute_pair_term = make_eval_pair_pot(
        kernel_fn=coulomb_kernel,
        pbc=(False,) * n_dim,
        per_particle=True,
    )
    return 2 * compute_pair_term(positions, charges) / charges


def calc_stress_virial(
    positions: ArrayLike, forces: ArrayLike, cell: ArrayLike
) -> Array:
    """Calculate stress tensor from virial

    Args:
        positions: Array of particle positions, shape `(n_particles, n_dim)`
        forces: Array of forces, shape `(n_particles, n_dim)`
        cell: Array representing cell, shape `(n_dim, n_dim)`

    Returns:
        Stress in 6-component format.
    """
    volume = jnp.linalg.det(cell)
    (i, j) = inds_matrix_to_six_component_stress
    dotproducts = jax.vmap(jnp.dot, in_axes=(1, 1))(
        positions[:, i], forces[:, j]
    )
    return -dotproducts / volume


def calc_nonperiodic_reference_results(
    positions: ArrayLike, charges: ArrayLike, cell: ArrayLike
) -> tuple[Array, Array, Array, Array]:
    """Calculate various exact reference for a non-periodic system

    Args:
        positions: Array of particle positions, shape `(n_particles, n_dim)`
        charges: Array of particle charges, shape `(n_particles,)`
        cell: Array representing cell, shape `(n_dim, n_dim)`

    Returns:
        Energy, forces, charge gradient, stress tensor
    """
    # The following is less likely to run out of memory than putting the
    # whole function under a single jit
    energy = jax.jit(calc_energy_nonperiodic_direct_summation)(
        positions, charges
    )
    forces = jax.jit(calc_forces_nonperiodic_direct_summation)(
        positions, charges
    )
    chargegrad = jax.jit(calc_chargegrad_nonperiodic_direct_summation)(
        positions, charges
    )
    stress = jax.jit(calc_stress_virial)(positions, forces, cell)
    return energy, forces, chargegrad, stress
