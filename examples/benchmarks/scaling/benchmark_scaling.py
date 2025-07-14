"""Demonstration of scaling of the MSM implementation with particle number."""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

try:
    # TODO: This might not actually solve the problem in newer JAX versions.
    #  Check which exception is actually raised in newer versions when running
    #  out of memory, it might not be the same one!
    from jaxlib._jax import XlaRuntimeError
except ModuleNotFoundError:
    # To work with older JAX versions
    from jaxlib.xla_extension import XlaRuntimeError

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.core.shortrange import _gen_supercell, make_eval_pair_pot
from msmjax.utils.benchmarking import (
    build_duplicate_free_neighborlists,
    make_timed_eval,
    path_input_structures,
)

# TODO: Explicitly compute as the average particle spacing instead?
LEVEL_ONE_SPACING = 1.0

# TODO: energy, other quantities?
QUANTITY = "forces"


def coulomb_kernel(r):
    return 1.0 / r


def calc_nonperiodic_ref_energy(positions, charges):
    # TODO: name
    # TODO: move to utils? (used here, in cost-vs-accuracy benchmark, and in
    #  tests/data/generate_reference_results.ipynb)
    n_dim = positions.shape[1]
    compute_pair_term = make_eval_pair_pot(
        kernel_fn=coulomb_kernel, pbc=(False,) * n_dim
    )
    return compute_pair_term(positions, charges)


def calc_nonperiodic_ref_forces(positions, charges):
    # TODO: name
    # TODO: move to utils? (used here, in cost-vs-accuracy benchmark, and in
    #  tests/data/generate_reference_results.ipynb)
    return -jax.grad(calc_nonperiodic_ref_energy, argnums=0)(
        positions, charges
    )


exact_nonperiodic_evaluation_fns = {
    "energy": calc_nonperiodic_ref_energy,
    "forces": calc_nonperiodic_ref_forces,
}


def iter_structures(double_precision: bool = False):
    for repeats, unrepeated_particle_nums in zip(
        [None, 2, 3, 4],
        [
            [3000, 6000, 9000, 12000, 15000],
            [2500, 3500, 4500, 6000, 8000, 10000],
            [4000, 5000, 7000, 9000, 12000, 15000],
            [8000, 10000, 12000, 15000],
        ],
    ):
        for n_particles_original in unrepeated_particle_nums:
            structures = onp.load(
                path_input_structures
                / f"structures_{n_particles_original}.npz"
            )
            if double_precision:
                pos = jax.device_put(
                    structures["positions"][0].astype(onp.float64)
                )
                chg = jax.device_put(
                    structures["charges"][0].astype(onp.float64)
                )
                cell = jax.device_put(
                    structures["cells"][0].astype(onp.float64)
                )
            else:
                pos = jax.device_put(structures["positions"][0])
                chg = jax.device_put(structures["charges"][0])
                cell = jax.device_put(structures["cells"][0])
            if repeats is not None:
                pos, chg, cell = _gen_supercell(
                    pos, chg, cell, supercell_diag=[repeats] * 3
                )
            yield pos, chg, cell


msmparser = ArgumentParser(add_help=False)
msmparser.add_argument(
    "--cutoff",
    type=float,
    required=True,
    help="Level-zero cutoff radius of MSM.",
)
msmparser.add_argument(
    "-p", type=int, required=True, help="Interpolation order."
)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Demonstration of scaling of the MSM implementation "
        "with particle number."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Output directory. Existing outputs will not be overwritten.",
    )
    parser.add_argument(
        "--jax_enable_x64",
        action="store_true",
        default=False,
        help="Flag indicating that double precision should be used",
    )

    subparsers = parser.add_subparsers(
        dest="algo",
        help="The algorithm to benchmark.",
    )
    parser_nonperiodic_exact = subparsers.add_parser(
        "nonperiodic-exact",
        help="Do not use MSM, but exact all-pairs evaluation, "
        "for comparison. Only possible with non-periodic boundary conditions.",
    )
    parser_nonperiodic_msm = subparsers.add_parser(
        "nonperiodic-msm",
        parents=[msmparser],
        help="Benchmark MSM with non-periodic boundary conditions.",
    )
    parser_periodic_msm = subparsers.add_parser(
        "periodic-msm",
        parents=[msmparser],
        help="Benchmark MSM with periodic boundary conditions.",
    )

    cmd_args = parser.parse_args()
    if cmd_args.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("- Running JAX in double-precision mode.")
    baseoutdir = Path(cmd_args.outdir)
    baseoutdir.mkdir(parents=True)
    resultsfile = baseoutdir / "results.csv"

    for pos, chg, cell in iter_structures(
        double_precision=cmd_args.jax_enable_x64
    ):
        n_particles = pos.shape[0]
        print(f"- n_particles = {n_particles}")

        if cmd_args.algo == "nonperiodic-exact":
            fn = exact_nonperiodic_evaluation_fns[QUANTITY]
        else:
            if cmd_args.algo == "nonperiodic-msm":
                pbc = (False, False, False)
            elif cmd_args.algo == "periodic-msm":
                pbc = (True, True, True)
            level_zero_cutoff = cmd_args.cutoff
            p = cmd_args.p
            msm_params = set_up_msm_params(
                cell=cell,
                level_one_spacings=LEVEL_ONE_SPACING,
                level_zero_cutoff=level_zero_cutoff,
                p=p,
                pbc=pbc,
                cell_mode="ortho",
                dynamic_cell=False,
                n_particles=n_particles,
                use_neighborlist=True,
                neighborlist_prefactor=1.0,  # duplicate-free neighbor list
            )
            msm_evaluation_fns = create_msm(msm_params)
            fn = msm_evaluation_fns[QUANTITY]

        try:
            # TODO: Add "repeat" and "number" as command-line args?
            timing_fn = make_timed_eval(fn, repeat=10, number=15)
            if cmd_args.algo == "nonperiodic-exact":
                time, _ = timing_fn(pos, chg)
            else:
                neighborlist = build_duplicate_free_neighborlists(
                    [pos], [cell], level_zero_cutoff, pbc=pbc
                )[0]
                time, _ = timing_fn(pos, chg, neighborlist=neighborlist)
        except XlaRuntimeError as e:
            if "out of memory" in str(e).lower():
                print("- Out of memory: skipping the rest of the loop.")
                break
            else:
                raise e

        print(f"- time = {time * 1000:.2f} ms")

        outdata = {
            "n_particles": n_particles,
            "quantity": QUANTITY,
            "time": time,
        }
        if cmd_args.algo in ["nonperiodic-msm", "periodic-msm"]:
            outdata["level_zero_cutoff"] = level_zero_cutoff
            outdata["p"] = p
        results_tmp = pd.DataFrame(data=outdata, index=[0])
        print(f"- Writing results to {resultsfile}.")
        if not resultsfile.is_file():
            results_tmp.to_csv(resultsfile, index=False, mode="w")
        else:
            results_tmp.to_csv(
                resultsfile, index=False, mode="a", header=False
            )

        print()

    print()

    results = pd.read_csv(resultsfile)
    particle_numbers = results["n_particles"].values
    times = results["time"].values
    fig, ax = plt.subplots()
    ax.set_xlabel("Number of particles")
    ax.set_ylabel("Time / ms")
    ax.scatter(particle_numbers, times * 1000)
    # TODO: legend?
    ax.set_xscale("log")
    ax.set_yscale("log")
    filename_without_suffix = f"scaling_{QUANTITY}_loglog"
    for suffix in [".png", ".pdf"]:
        outfile_plot = baseoutdir / (filename_without_suffix + suffix)
        print(f"- Saving plot to: {outfile_plot}")
        fig.savefig(outfile_plot)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    filename_without_suffix = f"scaling_{QUANTITY}"
    for suffix in [".png", ".pdf"]:
        outfile_plot = baseoutdir / (filename_without_suffix + suffix)
        print(f"- Saving plot to: {outfile_plot}")
        fig.savefig(outfile_plot)
