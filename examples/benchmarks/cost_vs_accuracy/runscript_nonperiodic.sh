#!/bin/bash

python benchmark_cost_vs_accuracy.py \
  --structuretype nonperiodic \
  --quantity energy forces charge_gradient stress_diag stress_all \
  --outdir out/nonperiodic
