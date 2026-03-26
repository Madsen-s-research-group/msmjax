import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from pathlib import Path

import jax
import numpy as onp
import pandas as pd
from helpers_benchmark import (
    iter_structures_and_forces,
    set_up_ewald_jaxmd_dynalpha,
    set_up_pme_jaxmd_dynalpha,
)
from jax_md import partition
from natsort import natsorted

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.utils.benchmarking import (
    build_duplicate_free_neighborlists,
    calc_relative_rmse,
    make_timed_eval,
)

PBC = (True,) * 3
# Use a neighbor list that is only as large as necessary:
NEIGHBOR_KWARGS_JAXMD = {
    "capacity_multiplier": 1.0,
    "format": partition.NeighborListFormat.OrderedSparse,
}


def timing_loop_ewald(
    parameters: pd.DataFrame,
    side_length,
    positions,
    charges,
    reference_forces,
    target_accuracy,
    repeat=5,
    number=10,
):
    total = len(parameters)

    tmp_params = parameters.sort_values(
        by=["cutoff", "g_max", "alpha"], inplace=False
    )
    results_df = tmp_params.copy()
    results_df["relative_rmse"] = onp.nan
    results_df["time"] = onp.nan
    for _, par_lvl_1 in tmp_params.groupby(["cutoff"]):
        cutoff = par_lvl_1["cutoff"].iloc[0]
        # Construct neighbor list for the current cutoff.
        # We do this on CPU to prevent running out of GPU memory. While
        # this is a bit slower, it does not matter because it is not
        # included in the timing.
        with jax.default_device(jax.devices("cpu")[0]):
            nbr_fns, _ = set_up_ewald_jaxmd_dynalpha(
                side_length=side_length,
                cutoff=cutoff,
                g_max=5.0,  # dummy value (doesn't affect neighbor list)
                **NEIGHBOR_KWARGS_JAXMD,
            )
            nbl = nbr_fns.allocate(positions)
        for _, par_lvl_2 in par_lvl_1.groupby(["g_max"]):
            g_max = par_lvl_2["g_max"].iloc[0]
            _, eval_fns = set_up_ewald_jaxmd_dynalpha(
                side_length=side_length,
                cutoff=cutoff,
                g_max=g_max,
                **NEIGHBOR_KWARGS_JAXMD,
            )
            calc_forces = jax.jit(eval_fns["forces"])
            try:
                for alpha in par_lvl_2["alpha"].values:
                    cond = (
                        (results_df["cutoff"] == cutoff)
                        & (results_df["g_max"] == g_max)
                        & (results_df["alpha"] == alpha)
                    )
                    iteration_counter = onp.where(cond)[0][0] + 1
                    print(
                        f"- cutoff = {cutoff:.2f}, g_max = {g_max:.2f}, "
                        f"alpha = {alpha:.3f} "
                        f"(parameter combination {iteration_counter}/{total})"
                    )
                    call_args = {
                        "positions": positions,
                        "charges": charges,
                        "neighborlist": nbl,
                        "alpha": alpha,
                    }
                    timed_eval = make_timed_eval(calc_forces, repeat, number)
                    exec_time, output = timed_eval(**jax.device_put(call_args))
                    acc = calc_relative_rmse(output, reference_forces)
                    results_df.loc[cond, "relative_rmse"] = acc
                    results_df.loc[cond, "time"] = exec_time
                    print(f"  accuracy = {acc:.5f}, time = {exec_time:.4f}")
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(
                        "- Out of memory. Skipping any remaining "
                        "cutoff values."
                    )
                    print()
                    break
                else:
                    raise e
        df_ok_up_to_cutoff = results_df[
            (results_df["relative_rmse"] <= target_accuracy)
            & (results_df["cutoff"] <= cutoff)
        ]
        inds_min_time_by_cutoff = (
            df_ok_up_to_cutoff.groupby("cutoff")["time"].idxmin().array
        )
        min_times_by_cutoff = df_ok_up_to_cutoff.loc[
            inds_min_time_by_cutoff, "time"
        ]
        try:
            last_three_times = min_times_by_cutoff.to_numpy()[-3:]
            if (last_three_times[0] <= last_three_times[1]) & (
                last_three_times[1] <= last_three_times[2]
            ):
                print(
                    "  -> No improvement during the last two cutoffs. Ending search."
                )
                break
        except IndexError:
            continue

    # Restore the order of rows in the input dataframe (assuming the index
    # was not out of order to begin with)
    return results_df.sort_index()


def timing_loop_pme(
    parameters: pd.DataFrame,
    side_length,
    positions,
    charges,
    reference_forces,
    target_accuracy,
    repeat=5,
    number=10,
):
    total = len(parameters)

    tmp_params = parameters.sort_values(
        by=["cutoff", "grid_points", "alpha"], inplace=False
    )
    results_df = tmp_params.copy()
    results_df["relative_rmse"] = onp.nan
    results_df["time"] = onp.nan
    for _, par_lvl_1 in tmp_params.groupby(["cutoff"]):
        cutoff = par_lvl_1["cutoff"].iloc[0]
        # Construct neighbor list for the current cutoff.
        # We do this on CPU to prevent running out of GPU memory. While
        # this is a bit slower, it does not matter because it is not
        # included in the timing.
        with jax.default_device(jax.devices("cpu")[0]):
            nbr_fns, _ = set_up_pme_jaxmd_dynalpha(
                side_length=side_length,
                cutoff=cutoff,
                grid_points=2,  # dummy value (doesn't affect neighbor list)
                **NEIGHBOR_KWARGS_JAXMD,
            )
            nbl = nbr_fns.allocate(positions)
        for _, par_lvl_2 in par_lvl_1.groupby(["grid_points"]):
            grid_points = par_lvl_2["grid_points"].iloc[0]
            _, eval_fns = set_up_pme_jaxmd_dynalpha(
                side_length=side_length,
                cutoff=cutoff,
                grid_points=grid_points,
                **NEIGHBOR_KWARGS_JAXMD,
            )
            calc_forces = jax.jit(eval_fns["forces"])
            try:
                for alpha in par_lvl_2["alpha"].values:
                    cond = (
                        (results_df["cutoff"] == cutoff)
                        & (results_df["grid_points"] == grid_points)
                        & (results_df["alpha"] == alpha)
                    )
                    iteration_counter = onp.where(cond)[0][0] + 1
                    print(
                        f"- cutoff = {cutoff:.2f}, grid_points = {grid_points}, "
                        f"alpha = {alpha:.3f} "
                        f"(parameter combination {iteration_counter}/{total})"
                    )
                    call_args = {
                        "positions": positions,
                        "charges": charges,
                        "neighborlist": nbl,
                        "alpha": alpha,
                    }
                    timed_eval = make_timed_eval(calc_forces, repeat, number)
                    exec_time, output = timed_eval(**jax.device_put(call_args))
                    acc = calc_relative_rmse(output, reference_forces)
                    results_df.loc[cond, "relative_rmse"] = acc
                    results_df.loc[cond, "time"] = exec_time
                    print(f"  accuracy = {acc:.5f}, time = {exec_time:.4f}")
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(
                        "- Out of memory. Skipping any remaining "
                        "(smaller, more expensive) grid spacings."
                    )
                    print()
                    break
                else:
                    raise e
        df_ok_up_to_cutoff = results_df[
            (results_df["relative_rmse"] <= target_accuracy)
            & (results_df["cutoff"] <= cutoff)
        ]
        inds_min_time_by_cutoff = (
            df_ok_up_to_cutoff.groupby("cutoff")["time"].idxmin().array
        )
        min_times_by_cutoff = df_ok_up_to_cutoff.loc[
            inds_min_time_by_cutoff, "time"
        ]
        try:
            last_three_times = min_times_by_cutoff.to_numpy()[-3:]
            if (last_three_times[0] <= last_three_times[1]) & (
                last_three_times[1] <= last_three_times[2]
            ):
                print(
                    "  -> No improvement during the last two cutoffs. "
                    "Ending search."
                )
                break
        except IndexError:
            continue

    # Restore the order of rows in the input dataframe (assuming the index
    # was not out of order to begin with)
    return results_df.sort_index()


def timing_loop_msm(
    parameters: pd.DataFrame,
    side_length,
    positions,
    charges,
    reference_forces,
    target_accuracy,
    repeat=5,
    number=10,
):
    total = len(parameters)

    cell = onp.diag([side_length] * 3)

    tmp_params = parameters.sort_values(
        by=["level_zero_cutoff", "level_one_spacings", "p"],
        ascending=[True, False, True],
        inplace=False,
    )
    results_df = tmp_params.copy()
    results_df["relative_rmse"] = onp.nan
    results_df["time"] = onp.nan
    for _, par_lvl_1 in tmp_params.groupby(["level_zero_cutoff"]):
        cutoff = par_lvl_1["level_zero_cutoff"].iloc[0]
        print(f"- Current cutoff = {cutoff:.2f}")
        # Construct neighbor list for the current cutoff.
        nbl = build_duplicate_free_neighborlists(
            [pos], [cell], cutoff, pbc=PBC
        )[0]
        for _, par_lvl_2 in par_lvl_1.groupby(["p"]):
            p = par_lvl_2["p"].iloc[0]
            for spacing in par_lvl_2["level_one_spacings"].values:
                msm_params = set_up_msm_params(
                    cell=cell,
                    level_one_spacings=spacing,
                    level_zero_cutoff=cutoff,
                    p=p,
                    pbc=PBC,
                    cell_mode="ortho",
                    dynamic_cell=False,
                    use_neighborlist=True,
                    neighborlist_prefactor=1.0,
                )
                eval_fns = create_msm(msm_params)
                calc_forces = jax.jit(eval_fns["forces"])
                cond = (
                    (results_df["level_zero_cutoff"] == cutoff)
                    & (results_df["level_one_spacings"] == spacing)
                    & (results_df["p"] == p)
                )
                iteration_counter = onp.where(cond)[0][0] + 1
                print(
                    f"- level_zero_cutoff = {cutoff:.2f}, "
                    f"level_one_spacings = {spacing:.2f}, p = {p} "
                    f"(parameter combination {iteration_counter}/{total})"
                )
                call_args = {
                    "positions": positions,
                    "charges": charges,
                    "neighborlist": nbl,
                }
                try:
                    timed_eval = make_timed_eval(calc_forces, repeat, number)
                    exec_time, output = timed_eval(**jax.device_put(call_args))
                    acc = calc_relative_rmse(output, reference_forces)
                    results_df.loc[cond, "relative_rmse"] = acc
                    results_df.loc[cond, "time"] = exec_time
                    print(f"  accuracy = {acc:.5f}, time = {exec_time:.4f}")
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        print(
                            "- Out of memory. Skipping any remaining "
                            "(smaller, more expensive) grid spacings."
                        )
                        print()
                        break
                    else:
                        raise e
        df_ok_up_to_cutoff = results_df[
            (results_df["relative_rmse"] <= target_accuracy)
            & (results_df["level_zero_cutoff"] <= cutoff)
        ]
        inds_min_time_by_cutoff = (
            df_ok_up_to_cutoff.groupby("level_zero_cutoff")["time"]
            .idxmin()
            .array
        )
        min_times_by_cutoff = df_ok_up_to_cutoff.loc[
            inds_min_time_by_cutoff, "time"
        ]
        try:
            last_three_times = min_times_by_cutoff.to_numpy()[-3:]
            if (last_three_times[0] <= last_three_times[1]) & (
                last_three_times[1] <= last_three_times[2]
            ):
                print(
                    "  -> No improvement during the last two cutoffs. Ending search."
                )
                break
        except IndexError:
            continue

    # Restore the order of rows in the input dataframe (assuming the index
    # was not out of order to begin with)
    return results_df.sort_index()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--method", required=True, choices=["msm", "ewald", "pme"]
    )
    parser.add_argument("--target_accuracy", required=True, type=float)
    parser.add_argument("--indir_params", required=True)
    parser.add_argument("--outdir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--max_n_particles", type=int)
    group.add_argument("--request_n_particles", type=int, nargs="+")
    args = parser.parse_args()
    indir_params = Path(args.indir_params)
    outdir = Path(args.outdir)
    if args.method == "msm":
        timing_loop = timing_loop_msm
    elif args.method == "ewald":
        timing_loop = timing_loop_ewald
    elif args.method == "pme":
        timing_loop = timing_loop_pme
    else:
        raise ValueError(f"Invalid method: {args.method}")

    outdir.mkdir(parents=True)
    resultsfile = outdir / f"results.csv"

    if args.max_n_particles is not None:
        iterator = iter_structures_and_forces(
            max_n_particles=args.max_n_particles
        )
    elif args.request_n_particles is not None:
        iterator = iter_structures_and_forces(
            request_n_particles=args.request_n_particles
        )
    else:
        param_files = natsorted(indir_params.glob(f"trial_params_*.csv"))
        filename_max_n_wo_suffix = param_files[-1].with_suffix("").parts[-1]
        max_n_particles = int(filename_max_n_wo_suffix.split("_")[-1])
        iterator = iter_structures_and_forces(max_n_particles=max_n_particles)

    list_of_n_particles = []
    list_of_dfs = []
    for pos, chg, cll, forces, _ in iterator:
        n_particles = pos.shape[0]
        print("#" * 80)
        print(f"- n_particles = {n_particles}")
        print("#" * 80)
        df_trial_params = pd.read_csv(
            indir_params / f"trial_params_{n_particles}.csv",
            index_col=0,
        )
        params_and_results_df = timing_loop(
            parameters=df_trial_params,
            side_length=cll[0, 0],
            positions=pos,
            charges=chg,
            reference_forces=forces,
            target_accuracy=args.target_accuracy,
        )

        params_and_results_df.insert(0, "n_particles", n_particles)
        params_and_results_df.to_csv(
            outdir / f"all_param_sets_{n_particles}.csv", index=False
        )

        df_clean = params_and_results_df.dropna()
        if len(df_clean) == 0:
            print(
                "- Not a single timing and accuracy evaluation succeeded at "
                "the current number of particles, indicating out of memory "
                "-> Skipping remaining (larger) structures."
            )
            break
        df_clean = df_clean[df_clean["relative_rmse"] <= args.target_accuracy]
        if len(df_clean) == 0:
            print(
                "- No parameter combination achieved the target accuracy "
                "at the current number of particles."
            )
            continue
        fastest_valid_params = df_clean.iloc[[df_clean["time"].argmin()]]
        print(f"- Writing results to {resultsfile}.")
        if not resultsfile.is_file():
            fastest_valid_params.to_csv(resultsfile, index=False, mode="w")
        else:
            fastest_valid_params.to_csv(
                resultsfile, index=False, mode="a", header=False
            )
