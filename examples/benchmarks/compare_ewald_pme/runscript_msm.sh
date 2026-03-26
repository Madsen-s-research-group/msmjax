#!/bin/bash

python -W "ignore" time_algo.py \
  --method msm \
  --target_accuracy 1.e-3 \
  --indir_params prepared_data/trial_params/msm_target-accuracy-0.001 \
  --outdir out/msm_target-accuracy-0.001
