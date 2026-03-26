#!/bin/bash

python -W "ignore" time_algo.py \
  --method pme \
  --target_accuracy 1.e-3 \
  --indir_params prepared_data/trial_params/pme_target-accuracy-0.001 \
  --outdir out/pme_target-accuracy-0.001
