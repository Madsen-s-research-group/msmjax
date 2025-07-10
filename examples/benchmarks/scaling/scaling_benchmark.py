"""Analyze scaling with number of particles."""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matscipy.neighbours
import numpy as onp
import pandas as pd
from tqdm import tqdm

from msmjax.calculators import create_msm, set_up_msm_params
from msmjax.core.shortrange import _gen_supercell
from msmjax.utils.benchmarking import (
    calc_relative_rmse,
    make_timed_eval,
    path_input_structures,
)

# TODO: Explicitly compute as the average particle spacing instead?
LEVEL_ONE_SPACING = 1.0


def structure_generator():
    for repeats, unrepeated_particle_nums in zip(
        [None, 2, 3],
        [
            [3000, 6000, 9000, 12000, 15000],
            [2500, 3500, 4500, 6000, 8000, 10000],
            [4000, 5000, 7000, 9000, 12000, 15000],
        ],
    ):
        for n_particles_original in unrepeated_particle_nums:
            structures = onp.load(
                path_input_structures
                / f"structures_{n_particles_original}.npz"
            )
            # TODO: .astype(onp.float64)?
            # TODO: jax.device_put (where?)
            pos = jax.device_put(structures["positions"][0])
            chg = jax.device_put(structures["charges"][0])
            cell = jax.device_put(structures["cells"][0])
            if repeats is not None:
                pos, chg, cell = _gen_supercell(
                    pos, chg, cell, supercell_diag=[repeats] * 3
                )
            yield pos, chg, cell


if __name__ == "__main__":
    pass  # TODO

    for pos, chg, cell in structure_generator():
        print(pos.shape)
