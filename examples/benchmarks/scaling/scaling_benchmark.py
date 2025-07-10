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

# particle_nums_no_replicate = onp.arange(3000, 15001, 3000)
# particle_nums_replicate_2x2x2 = (
#     onp.array([2500, 3500, 4500, 6000, 8000, 10000]) * 2**3
# )
# particle_nums_replicate_3x3x3 = (
#     onp.array([4000, 5500, 7000, 9000, 12000, 15000]) * 3**3
# )
#
# particle_nums_all = onp.concatenate(
#     [
#         particle_nums_no_replicate,
#         particle_nums_replicate_2x2x2,
#         particle_nums_replicate_3x3x3,
#     ]
# )

# supercell_diags_all =     # TODO


def structure_generator():
    particle_nums_no_replicate = onp.arange(3000, 15001, 3000)
    particle_nums_replicate_2x2x2 = onp.array(
        [2500, 3500, 4500, 6000, 8000, 10000]
    )
    particle_nums_replicate_3x3x3 = onp.array(
        [4000, 5500, 7000, 9000, 12000, 15000]
    )


if __name__ == "__main__":
    pass  # TODO

    # print(particle_nums_all)
    #
    # print()
    #
    # mygen = (n for n in particle_nums_all)
    #
    # for n_particles in mygen:
    #     print(n_particles)
