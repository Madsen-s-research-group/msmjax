#!/bin/bash

python benchmark_scaling.py --outdir out/periodic-msm_rcut-3.0_p-4 periodic-msm --cutoff 3.0 -p 4
python benchmark_scaling.py --outdir out/periodic-msm_rcut-6.0_p-6 periodic-msm --cutoff 6.0 -p 6
