import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import json
import timeit
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import jax
import numpy as onp
from natsort import natsorted

from msmjax.calculators import MSMParams, create_msm, set_up_msm_params
from msmjax.utils.benchmarking import path_input_structures

# MSM cutoff to grid spacing ratio parameter
# TODO: command-line arg?
ALPHA = 3.0


def make_model_timing_fn(
    params: MSMParams,
    quantity: str,
    repeat: int = 10,
    number: int = 100,
    **kwargs,  # TODO: more informative variable name (clarifying that they will be passed to create_msm)
) -> Callable:
    """Set up an MSM model, return a function to time it on one structure,
    and a dictionary of model information."""
    evaluation_fns = create_msm(params, **kwargs)

    def compute_energy(positions, charges, cell):
        return evaluation_fns["energy"](positions, charges, cell)

    if quantity == "energy":
        target_fn = compute_energy
    elif quantity == "dr":
        target_fn = jax.grad(compute_energy, argnums=0)
    elif quantity == "dq":
        target_fn = jax.grad(compute_energy, argnums=1)
    elif quantity == "energy_and_dr":
        target_fn = jax.value_and_grad(compute_energy, argnums=0)
    elif quantity == "energy_and_dq":
        target_fn = jax.value_and_grad(compute_energy, argnums=1)
    elif quantity == "energy_and_dr_and_dq":
        target_fn = jax.value_and_grad(compute_energy, argnums=(0, 1))
    else:
        print("Illegal option")

    target_fn = jax.jit(target_fn)

    def time_model_eval(positions, charges, cell) -> float:
        """Time an MSM model on one structure using ``timeit.repeat()`` and
        return the minimum value out of ``repeat`` loops."""
        if quantity in [
            "energy_and_dr",
            "energy_and_dq",
            "energy_and_dr_and_dq",
        ]:
            # If the output is a container type, we cannot call
            # block_until_ready() on it directly, but first need to flatten
            # it down to one of the leaf arrays.
            fn_to_time = lambda: jax.tree.flatten(
                target_fn(positions, charges, cell)
            )[0][0].block_until_ready()
        else:
            fn_to_time = lambda: target_fn(
                positions, charges, cell
            ).block_until_ready()

        # Call once to ensure jit-compilation
        fn_to_time()

        times_per_loop = timeit.repeat(
            fn_to_time, repeat=repeat, number=number
        )
        mean_times_per_call = onp.array(times_per_loop) / number

        return min(mean_times_per_call)

    return time_model_eval


if __name__ == "__main__":
    parser = parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory. If it exists, it will not be overwritten.",
    )
    parser.add_argument(
        "--jax_enable_x64",
        action="store_true",
        default=False,
        help="Flag indicating that double precision should be used",
    )
    args = parser.parse_args()

    baseoutdir = Path(args.outdir)
    baseoutdir.mkdir(parents=True)

    # TODO: Starting at higher numbers of particles is necessary in non-periodic
    #  case, otherwise the cutoff condition will result in zero grid levels.
    #  But this is not necessary in periodic case -> still treat the same?
    # TODO: Related, but more general: The definition of the numbers of
    #  particles for which to run the benchmark could probably be streamlined
    # for npz_file in natsorted(path_input_structures.glob("structures_*.npz"))[
    #     3::4
    # ]:
    for n_particles in [
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ]:
        npz_file = path_input_structures / f"structures_{n_particles}.npz"
        structures = onp.load(npz_file)
        pos = structures["positions"][0]
        chg = structures["charges"][0]
        cell = structures["cells"][0]
        if args.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
            pos = pos.astype(onp.float64)
            chg = chg.astype(onp.float64)
            cell = cell.astype(onp.float64)
        pos = jax.device_put(pos)
        chg = jax.device_put(chg)
        cell = jax.device_put(cell)
        # n_particles = pos.shape[0]    # TODO

        print("-" * 80)
        print(f"{n_particles=}")
        print("-" * 80)

        volume = onp.linalg.det(cell)
        n_dim = pos.shape[1]
        avg_particle_spacing = (volume / n_particles) ** (1.0 / n_dim)
        level_one_spacing = avg_particle_spacing
        level_zero_cutoff = ALPHA * level_one_spacing

        # for pbc in [(False, False, False), (True, True, True)]: # TODO
        for pbc in [(True, True, True)]:
            # for custom_derivatives in [False, True]:  # TODO
            for custom_derivatives in [False, True]:
                for quantity in [
                    # "energy", # TODO
                    "dr",
                    # "dq", # TODO
                    # "energy_and_dr_and_dq",   # TODO
                ]:
                    label = "pbc-" + "".join([str(p)[0] for p in pbc])
                    if custom_derivatives:
                        label += "__customjvp"
                    else:
                        label += "__defaultgrad"
                    label += "__" + quantity
                    outdir = baseoutdir / label
                    outdir.mkdir(parents=True, exist_ok=True)

                    msm_params = set_up_msm_params(
                        cell=cell,
                        level_one_spacings=avg_particle_spacing,
                        level_zero_cutoff=level_zero_cutoff,
                        pbc=pbc,
                        cell_mode="ortho",
                        dynamic_cell=False,
                        n_particles=n_particles,
                    )
                    timing_fn = make_model_timing_fn(
                        params=msm_params,
                        part="longrange",
                        use_custom_derivatives_for_longrange=custom_derivatives,
                        quantity=quantity,
                    )
                    min_time = timing_fn(pos, chg, cell)
                    print(label + ":", min_time * 1000, "ms")
                    msm_params.save_json(
                        outdir / f"msm_params_n_particles_{n_particles}.json"
                    )
                    with open(outdir / "times_vs_n_particles.txt", "a+") as f:
                        f.write(f"{n_particles:>5} {min_time:.6f}\n")
