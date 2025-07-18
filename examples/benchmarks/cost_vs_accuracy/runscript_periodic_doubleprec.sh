#!/bin/bash

python benchmark_cost_vs_accuracy.py --jax_enable_x64 --structuretype periodic --quantity energy forces --outdir out/periodic_doubleprec
