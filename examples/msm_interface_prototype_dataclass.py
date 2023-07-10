from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import jax_md


@dataclass
class MSM:
    cell: ...  # data structure to be decided (might be passed together with pbc as a combined data structure)
    pbc: ...  # data structure to be decided (might be passed together with cell as a combined data structure)
    n_particles: int  # neighbor list allocation needs some way of knowing number of particles
    cutoff_level_zero: float
    gridspacing_level_one: float
    interpolation_order: int
    mu: int
    highest_level: int

    def __post_init__(self):
        # Neighbor list allocation requires:
        # self.cell, self.pbc, self.n_particles, self.cutoff_level_zero
        self.shortrange_neighborlist_fn = jax_md.partition.neighbor_list(...)

        # Apart from the neighbor list functions, also creates or computes:
        # - kernel functions for all levels
        # - spline eval function
        # - restriction/prolongation coefficients
        # - kernel stencils
        # - other things I'm probably forgetting about at the moment

    def calculate_energy_contrib_shortrange(
        self, positions, charges
    ) -> jnp.ndarray:
        pass

    def calculate_force_contrib_shortrange(
        self, positions, charges
    ) -> jnp.ndarray:
        """Could also use grad for this"""
        pass

    def calculate_gridpotential_oneplus(
        self, positions, charges
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Besides the 'grid potential' e^{1+}, this also returns the indices of the grid points within spline range of each particle"""
        pass

    def calculate_energy_contrib_grids(
        self, positions, charges, gridpotential_oneplus, indices
    ) -> jnp.ndarray:
        pass

    def calculate_force_contrib_grids(
        self, positions, charges, gridpotential_oneplus, indices
    ) -> jnp.ndarray:
        pass

    #
    def __call__(
        self,
        positions: jnp.ndarray,
        charges: jnp.ndarray,
        shortrange_neighborlist: jax_md.partition.NeighborList,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jax_md.partition.NeighborList]:
        # short-range part
        shortrange_neighborlist = self.shortrange_neighborlist_fn.update(
            positions, shortrange_neighborlist
        )
        e_0 = self.calculate_energy_contrib_shortrange(
            positions, charges, shortrange_neighborlist
        )
        f_0 = self.calculate_force_contrib_shortrange(
            positions, charges, shortrange_neighborlist
        )

        # grid part
        gridpotential_oneplus, indices = self.calculate_gridpotential_oneplus(
            positions, charges
        )
        e_oneplus = self.calculate_energy_contrib_grids(
            positions, charges, gridpotential_oneplus, indices
        )
        f_oneplus = self.calculate_force_contrib_grids(
            positions, charges, gridpotential_oneplus, indices
        )

        # combine into total energy and forces
        energy = e_0 + e_oneplus
        forces = f_0 + f_oneplus

        return energy, forces, shortrange_neighborlist


if __name__ == "__main__":
    msm = MSM(
        cell=...,
        pbc=...,
        n_particles=1000,
        cutoff_level_zero=2.0,
        gridspacing_level_one=0.8,
        interpolation_order=4,
        mu=6,
        highest_level=10,
    )
    shortrange_neighborlist = msm.shortrange_neighborlist_fn.allocate(
        positions
    )
    for _ in n_steps:
        # could also be wrapped in lax.scan
        energy, forces, shortrange_neighborlist = msm(
            positions, charges, shortrange_neighborlist
        )
        positions = update_positions(...)
        charges = update_charges(...)
