#!/bin/bash

python run_benchmark.py --jax_enable_x64 --structuretype nonperiodic --quantity energy forces --outdir out/nonperiodic_doubleprec
