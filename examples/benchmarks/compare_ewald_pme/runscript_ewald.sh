#!/bin/bash

python -W "ignore" time_algo.py \
  --method ewald \
  --target_accuracy 1.e-3 \
  --indir_params prepared_data/trial_params/ewald_target-accuracy-0.001 \
  --outdir out/ewald_target-accuracy-0.001
