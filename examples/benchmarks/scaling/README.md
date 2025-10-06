Analysis of scaling of execution time with the number of particles.

Example results, obtained on an NVIDIA A40 GPU, using jax and jaxlib version 0.7, and CUDA toolkit version 12.9, are contained in `example_results/`.

To run the benchmarks yourself, execute the various wrappers ([`runscript_nonperiodic.sh`](runscript_nonperiodic.sh), [`runscript_nonperiodic_doubleprec.sh`](runscript_nonperiodic_doubleprec.sh), [`runscript_periodic.sh`](runscript_periodic.sh), [`runscript_periodic_doubleprec.sh`](runscript_periodic_doubleprec.sh)), which internally call the [`benchmark_scaling.py`](benchmark_scaling.py) script with the correct command-line arguments.
To combine the results for various MSM settings, periodicity/no periodicity, single/double precision into a single plot, the [`plot_combined.ipynb`](plot_combined.ipynb) is provided.
