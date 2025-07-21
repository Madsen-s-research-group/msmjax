#!/bin/bash

python benchmark_cost_vs_accuracy.py \
  --jax_enable_x64 \
  --structuretype slab-nozdipole \
  --quantity energy forces charge_gradient stress \
  --outdir out/slab-nozdipole
