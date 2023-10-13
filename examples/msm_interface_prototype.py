def set_up_msm(
    box,
    pbc,
    cutoff_level_zero,
    gridspacing_level_one,
    interpolation_order,
    mu,
    highest_level,
):
    """Create all closures needed in MSM energy and force evaluations.

    Args:
        box:
        pbc:
        cutoff_level_zero: Cutoff radius of the lowest-level (short-range) kernel.
        gridspacing_level_one: Spacing of the grid at level one.
        interpolation_order:
        mu:
        highest_level: Highest grid level

    Returns:
        Tuple containing:

            - jax-md NeighborListFns object for allocating and updating the
              neighbor list associated with the lowest-level kernel.
            - Function that calculates the lowest-level energy contribution
              from a given neighbor list and positions.
            - Function that calculates the lowest-level force contribution
              from a given neighbor list and positions.
            - Function that calculates the quantity $e^{1+}$
              (grid representation of total electrostatic potential generated
              by all kernels l > 1 combined).
            - Function that calculates contribution to the energy from grids.
            - Function that calculates contribution to the force from grids.

        For readability, one could further subsume the first three
        (=short-range-kernel-related) and last three (=grid-related) functions
        into individual tuples.

    """
    # Code for setting up MSM goes here...

    return (
        shortrange_neighborlist_fn,
        calculate_energy_contrib_shortrange,
        calculate_force_contrib_shortrange,
        calculate_gridpotential_oneplus,
        calculate_energy_contrib_grids,
        calculate_force_contrib_grids,
    )


if __name__ == "__main__":
    (
        shortrange_neighborlist_fn,
        calculate_energy_contrib_shortrange,
        calculate_force_contrib_shortrange,
        calculate_gridpotential_oneplus,
        calculate_energy_contrib_grids,
        calculate_force_contrib_grids,
    ) = set_up_msm(
        box=...,
        pbc=...,
        cutoff_level_zero=...,
        gridspacing_level_one=...,
        interpolation_order=...,
        mu=...,
        highest_level=...,
    )

    # allocate neighbor list for the short range part
    shortrange_neighborlist = shortrange_neighborlist_fn.allocate(positions)

    # run simulation loop (meant to represent something like MD)
    for _ in n_steps:
        # short-range part
        shortrange_neighborlist = shortrange_neighborlist_fn.update(
            positions, shortrange_neighborlist
        )
        e_0 = calculate_energy_contrib_shortrange(
            positions, charges, shortrange_neighborlist
        )
        f_0 = calculate_force_contrib_shortrange(
            positions, charges, shortrange_neighborlist
        )

        # grid part
        gridpotential_oneplus = calculate_gridpotential_oneplus(
            positions, charges
        )
        e_oneplus = calculate_energy_contrib_grids(
            positions, charges, gridpotential_oneplus
        )
        f_oneplus = calculate_force_contrib_grids(
            positions, charges, gridpotential_oneplus
        )

        # combine into total energy and forces
        e_electrostatic = e_0 + e_oneplus
        f_electrostatic = f_0 + f_oneplus

        # update positions, charges
        positions = update_positions(...)
        charges = update_charges(...)
