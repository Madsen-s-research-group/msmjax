#!/bin/bash

python benchmark_cost_vs_accuracy.py \
  --structuretype nonperiodic \
  --quantity energy forces charge_gradient stress \
  --outdir out/nonperiodic
