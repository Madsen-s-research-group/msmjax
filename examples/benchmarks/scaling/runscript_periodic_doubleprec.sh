#!/bin/bash

python benchmark_scaling.py --jax_enable_x64 --outdir out/periodic-msm_rcut-3.0_p-4_doubleprec periodic-msm --cutoff 3.0 -p 4
python benchmark_scaling.py --jax_enable_x64 --outdir out/periodic-msm_rcut-6.0_p-6_doubleprec periodic-msm --cutoff 6.0 -p 6
