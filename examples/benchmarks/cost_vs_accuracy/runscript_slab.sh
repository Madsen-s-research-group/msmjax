#!/bin/bash

python benchmark_cost_vs_accuracy.py \
  --structuretype slab \
  --quantity energy forces charge_gradient stress \
  --outdir out/slab
