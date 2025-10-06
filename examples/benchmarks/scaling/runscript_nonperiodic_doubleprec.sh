#!/bin/bash

python benchmark_scaling.py --jax_enable_x64 --outdir out/nonperiodic-exact_doubleprec nonperiodic-exact
python benchmark_scaling.py --jax_enable_x64 --outdir out/nonperiodic-msm_rcut-3.0_p-4_doubleprec nonperiodic-msm --cutoff 3.0 -p 4
python benchmark_scaling.py --jax_enable_x64 --outdir out/nonperiodic-msm_rcut-6.0_p-6_doubleprec nonperiodic-msm --cutoff 6.0 -p 6
