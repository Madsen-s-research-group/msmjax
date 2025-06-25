#!/bin/bash

python run_benchmark.py --jax_enable_x64 --structuretype periodic --quantity energy forces --outdir out/periodic_doubleprec
