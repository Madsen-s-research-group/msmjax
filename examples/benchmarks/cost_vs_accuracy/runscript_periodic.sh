#!/bin/bash

python benchmark_cost_vs_accuracy.py \
  --structuretype periodic \
  --quantity energy forces charge_gradient stress \
  --outdir out/periodic
