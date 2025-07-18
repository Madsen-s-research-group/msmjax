#!/bin/bash

python benchmark_cost_vs_accuracy.py --jax_enable_x64 --structuretype nonperiodic --quantity energy forces --outdir out/nonperiodic_doubleprec
