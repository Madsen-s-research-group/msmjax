#!/bin/bash

python benchmark_scaling.py --outdir out/nonperiodic-exact nonperiodic-exact
python benchmark_scaling.py --outdir out/nonperiodic-msm_rcut-3.0_p-4 nonperiodic-msm --cutoff 3.0 -p 4
python benchmark_scaling.py --outdir out/nonperiodic-msm_rcut-6.0_p-6 nonperiodic-msm --cutoff 6.0 -p 6
