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

from msmjax.benchmark_tools import get_metadata, path_input_structures
from msmjax.convenience import (
    set_up_kernels_grids_and_stencils,
    suggest_msm_params,
)
from msmjax.core.longrange import make_compute_u_oneplus, make_grid_pass_fn
from msmjax.gridops_multidim import create_all_grid_to_grid_ops

# MSM cutoff to grid spacing ratio parameter
ALPHA = 3.0


def make_model_timing_fn(
    cell,
    n_particles,
    pbc,
    msm_params_in: dict,
    use_custom_derivatives: bool,
    quantity: str,
    convolution_methods=None,
    repeat=10,
    number=100,
) -> tuple[Callable, dict]:
    """Set up an MSM model, return a function to time it on one structure,
    and a dictionary of model information."""
    ############################################################################
    # TODO: Begin of setup that might still be subject to API changes
    ############################################################################
    box_lengths = onp.diag(cell)
    msm_params_full = suggest_msm_params(
        box_lengths=box_lengths,
        pbc=pbc,
        n_particles=n_particles,
        **msm_params_in,
    )
    kernel_fns, grids, kernel_stencils = set_up_kernels_grids_and_stencils(
        box_lengths=box_lengths, pbc=pbc, **msm_params_full
    )
    grid_shapes = [None if g is None else g.shape for g in grids]
    kernel_stencil_shapes = [
        None if k is None else k.shape for k in kernel_stencils
    ]
    setup_info = {
        "box_lengths": onp.asarray(box_lengths).tolist(),
        "pbc": onp.asarray(pbc).tolist(),
        "n_particles": n_particles,
        "msm_params": msm_params_full,
        "convolution_methods": convolution_methods,
        "grid_shapes": grid_shapes,
        "kernel_stencil_shapes": kernel_stencil_shapes,
    }

    restriction_fns, prolongation_fns, interaction_fns = (
        create_all_grid_to_grid_ops(grids, convolution_methods)
    )
    grid_pass_fn = make_grid_pass_fn(
        restriction_fns, prolongation_fns, interaction_fns
    )
    ############################################################################
    # TODO: End of setup that might still be subject to API changes
    ############################################################################

    _compute_u_oneplus = make_compute_u_oneplus(
        singleparticle_basis_fn_lvl_one=grids[
            1
        ].evaluate_bspline_basis_one_particle,
        grid_pass_fn=grid_pass_fn,
        grid_shape_lvl_one=grids[1].shape,
        use_custom_derivatives=use_custom_derivatives,
    )

    # The raw functions coming out of make_compute_u_oneplus take parameters
    # (positions, charges, kernel_stencils). Make the signature compatible with
    # what the timing function expects:

    def compute_energy(positions, charges, cell):
        return _compute_u_oneplus(positions, charges, kernel_stencils)

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

    return time_model_eval, setup_info


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

    metadata = get_metadata()
    with open(baseoutdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for npz_file in natsorted(path_input_structures.glob("structures_*.npz"))[
        ::3
    ]:
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
        n_particles = pos.shape[0]

        print("-" * 80)
        print(f"{n_particles=}")
        print("-" * 80)

        volume = onp.linalg.det(cell)
        n_dim = pos.shape[1]
        avg_particle_spacing = (volume / n_particles) ** (1.0 / n_dim)
        msm_params_in = {
            "level_one_gridspacing": avg_particle_spacing,
            "alpha": ALPHA,
        }

        for pbc in [(False, False, False), (True, True, True)]:
            for use_custom_derivatives in [False, True]:
                for quantity in [
                    "energy",
                    "dr",
                    "dq",
                    "energy_and_dr_and_dq",
                ]:
                    label = "pbc-" + "".join([str(p)[0] for p in pbc])
                    label += "__" + (
                        "customjvp"
                        if use_custom_derivatives
                        else "defaultgrad"
                    )
                    label += "__" + quantity
                    outdir = baseoutdir / label
                    outdir.mkdir(parents=True, exist_ok=True)

                    timing_fn, info = make_model_timing_fn(
                        cell=cell,
                        n_particles=n_particles,
                        pbc=pbc,
                        use_custom_derivatives=use_custom_derivatives,
                        quantity=quantity,
                        convolution_methods="scipy-fft",
                        msm_params_in=msm_params_in,
                    )
                    min_time = timing_fn(pos, chg, cell)
                    print(label + ":", min_time * 1000, "ms")
                    with open(
                        outdir / f"info_n_particles_{n_particles}.json", "w"
                    ) as f:
                        json.dump(info, f)
                    with open(outdir / "times_vs_n_particles.txt", "a+") as f:
                        f.write(f"{n_particles:>5} {min_time:.6f}\n")
