import ase.units
import jax
import jax.numpy as jnp
import numpy as onp
from ase.calculators.calculator import Calculator, all_changes

# LJ parameters for Argon taken from:
# John A. White, J. Chem. Phys. 22 November 1999; 111 (20): 9352–9356. https://doi.org/10.1063/1.479848
EPSILON_ELECTRONVOLT = 125.7 * ase.units.kB
SIGMA_ANGSTROM = 3.345


def f_c(r, r_cut, r_onset):
    """From github.com/jax-md/jax-md/blob/main/jax_md/energy.py"""
    inner = jnp.where(
        r < r_cut,
        (r_cut - r) ** 2
        * (r_cut + 2 * r - 3 * r_onset)
        / (r_cut - r_onset) ** 3,
        0,
    )
    return jnp.where(r < r_onset, 1, inner)


def lennard_jones_potential(
    r, sigma=1.0, epsilon=1.0, r_cut=None, r_onset=None
):
    # TODO: Name ("single" does not quite describe it, because broadcasting
    #  still allows passing an array of rs)
    # TODO: type KernelFn?
    """Evaluate Lennard-Jones potential for scalar distance arguments."""
    inv_scaled_r = sigma / r
    inv_scaled_r2 = inv_scaled_r * inv_scaled_r
    inv_scaled_r6 = inv_scaled_r2 * inv_scaled_r2 * inv_scaled_r2
    inv_scaled_r12 = inv_scaled_r6 * inv_scaled_r6
    energy = 4 * epsilon * (inv_scaled_r12 - inv_scaled_r6)
    if r_cut is None:
        return energy
    else:
        return f_c(r, r_cut=r_cut, r_onset=r_onset) * energy


class GenericWrapperCalculator(Calculator):
    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, energy_fn, forces_fn, stress_fn):
        self._calc_energy = jax.jit(energy_fn)
        self._calc_forces = jax.jit(forces_fn)
        self._calc_stress = jax.jit(stress_fn)

        super().__init__()

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        if "energy" in properties:
            res = self._calc_energy(
                positions=self.atoms.positions, cell=self.atoms.cell[...]
            )
            self.results["energy"] = float(res)
        if "forces" in properties:
            res = self._calc_forces(
                positions=self.atoms.positions, cell=self.atoms.cell[...]
            )
            self.results["forces"] = onp.array(res)
        if "stress" in properties:
            # TODO: Does this return the stress in correct format?
            #  (Probably yes: Looking at `Atoms.get_stress()`, it looks like 3x3
            #  arrays are automatically detected and converted to the Voigt
            #  6-component format.)
            # TODO: What about the `include_ideal_gas` argument of
            #  `Atoms.get_stress()`? Should this be included or not?
            #  (probably yes)
            res = self._calc_stress(
                positions=self.atoms.positions, cell=self.atoms.cell[...]
            )
            self.results["stress"] = onp.array(res)
