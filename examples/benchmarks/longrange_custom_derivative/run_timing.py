import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import timeit
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Tuple

import jax
import numpy as onp
from matplotlib import pyplot as plt

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import path_input_structures

# MSM cutoff to grid spacing ratio parameter
# TODO: command-line arg?
ALPHA = 3.0


def make_timed_eval(
    fn: Callable, repeat: int = 10, number: int = 100
) -> Callable:
    # TODO: Move this function to utils?
    jitted_fn = jax.jit(fn)

    def time_model_eval(*args, **kwargs) -> Tuple[float, Any]:
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory. Existing outputs will not be overwritten.",
    )
    parser.add_argument(
        "--jax_enable_x64",
        action="store_true",
        default=False,
        help="Flag indicating that double precision should be used",
    )
    cmd_args = parser.parse_args()

    baseoutdir = Path(cmd_args.outdir)
    baseoutdir.mkdir(parents=True)

    for n_particles in [500, 1500, 2500, 3500, 4500, 6000, 8000, 10000]:
        npz_file = path_input_structures / f"structures_{n_particles}.npz"
        structures = onp.load(npz_file)
        pos = structures["positions"][0]
        chg = structures["charges"][0]
        cell = structures["cells"][0]
        if cmd_args.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
            pos = pos.astype(onp.float64)
            chg = chg.astype(onp.float64)
            cell = cell.astype(onp.float64)
        pos = jax.device_put(pos)
        chg = jax.device_put(chg)
        cell = jax.device_put(cell)

        print("-" * 80)
        print(f"{n_particles=}")
        print("-" * 80)

        volume = onp.linalg.det(cell)
        n_dim = pos.shape[1]
        avg_particle_spacing = (volume / n_particles) ** (1.0 / n_dim)
        level_one_spacing = avg_particle_spacing
        level_zero_cutoff = ALPHA * level_one_spacing

        for pbc in [(False, False, False), (True, True, True)]:
            for custom_derivatives in [False, True]:
                for quantity in ["energy", "dr", "dq", "energy_and_dr_and_dq"]:
                    dirname = "pbc-" + "".join([str(p)[0] for p in pbc])
                    if quantity == "energy":
                        dirname += "__energy"
                        if custom_derivatives:
                            # No need to run the undifferentiated energy
                            # evaluation itself twice, in both the default
                            # and custom derivative branches of the loop.
                            continue
                    else:
                        if custom_derivatives:
                            dirname += "__customjvp" + "__" + quantity
                        else:
                            dirname += "__defaultgrad" + "__" + quantity
                    outdir = baseoutdir / dirname
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
                    msm_evaluation_fns = create_msm(
                        msm_params,
                        part="longrange",
                        use_custom_derivatives_for_longrange=custom_derivatives,
                    )
                    if quantity == "energy":
                        fn = msm_evaluation_fns["energy"]
                    elif quantity == "dr":
                        fn = msm_evaluation_fns["forces"]
                    elif quantity == "dq":
                        fn = msm_evaluation_fns["charge_gradient"]
                    elif quantity == "energy_and_dr_and_dq":
                        fn = jax.value_and_grad(
                            msm_evaluation_fns["energy"], argnums=(0, 1)
                        )
                    else:
                        raise ValueError("Unknown quantity")
                    timing_fn = make_timed_eval(fn)
                    min_time, _ = timing_fn(pos, chg)
                    print(f"{dirname}: {min_time * 1000:.2f} ms")
                    msm_params.save_json(
                        outdir / f"msm_params_n_particles_{n_particles}.json"
                    )
                    with open(outdir / "times_vs_n_particles.txt", "a+") as f:
                        f.write(f"{n_particles:>5} {min_time:.6f}\n")

    print()

    print("-" * 80)
    print(f"Making plots")
    print("-" * 80)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markerlist = ["o", "s", "D"]
    plt.rcParams["font.size"] = 14

    pbc_labels = ["pbc-FFF", "pbc-TTT"]
    labelmap_quantities = {
        "dr": r"$\nabla_i U^{1+}$",
        "dq": r"$\frac{\partial U^{1+}}{\partial q_i}$",
        "energy_and_dr_and_dq": r"$\left(U^{1+}, \, \nabla_i U^{1+}, \, \frac{\partial U^{1+}}{\partial q_i}\right)$",
    }
    labelmap_customderivs = {
        "energy": "$U^{1+}$",
        "defaultgrad": "default autodiff",
        "customjvp": "opt. derivative",
    }

    for lbl_pbc in pbc_labels:
        filename_without_suffix = (
            "timing_default_vs_custom_grad" + "__" + lbl_pbc
        )
        fig, axs = plt.subplots(
            nrows=3,
            ncols=1,
            sharex=True,
            figsize=(6.4, 7.4),
        )
        axs[-1].set_xlabel("number of particles / $10^3$")
        for (quantity_identifier, quantity_nice_label), ax in zip(
            labelmap_quantities.items(), axs
        ):
            ax.set_title(quantity_nice_label, x=0.05, y=0.75, ha="left")
            ax.set_ylabel("time / ms")

            for (deriv_identifier, deriv_nice_label), color, marker in zip(
                labelmap_customderivs.items(), default_colors, markerlist
            ):
                dirname = "__".join([lbl_pbc, deriv_identifier])
                if deriv_identifier != "energy":
                    dirname += "__" + quantity_identifier
                resultsdir = baseoutdir / dirname
                try:
                    nbs_particles = onp.loadtxt(
                        resultsdir / "times_vs_n_particles.txt",
                        usecols=0,
                        dtype=int,
                    )
                    times = onp.loadtxt(
                        resultsdir / "times_vs_n_particles.txt",
                        usecols=1,
                        dtype=float,
                    )
                except FileNotFoundError:
                    continue
                ax.plot(
                    nbs_particles / 1000,
                    times * 1000,
                    label=deriv_nice_label,
                    marker=marker,
                    color=color,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                )

            ax.set_xlim(0, ax.get_xlim()[1])
            ax.set_ylim(0, ax.get_ylim()[1])

        plt.subplots_adjust(
            left=0.13, right=0.99, bottom=0.09, top=0.925, hspace=0.075
        )
        fig.align_ylabels()
        axs[0].legend(
            ncols=3,
            handletextpad=0.25,
            columnspacing=0.5,
            bbox_to_anchor=(0.975, 1.3),
            bbox_transform=axs[0].transAxes,
        )

        for suffix in ["png", "pdf"]:
            filename_plot = filename_without_suffix + "." + suffix
            print(f"Saving plot to {filename_plot}")
            fig.savefig(baseoutdir / filename_plot)
