Analysis of the tradeoff between accuracy and computational cost as a function of MSM algorithm parameters.
Can be used for parameter tuning (possibly by adapting to specific systems and quantities of interest other than the included ones).

The example includes two different benchmark cases, a non-periodic system of medium size (10000 particles), and a system under periodic boundary conditions with 100 particles in the unit cell.
A set of prepared reference results against which the MSM-computed results are compared is contained in `reference_data/`.
The quantities considered are energies, forces, gradients w.r.t. charges, and stresses.

Example results, obtained on an NVIDIA A40 GPU, using jax and jaxlib version 0.7, and CUDA toolkit version 12.9, are contained in `example_results/`.

To run the benchmarks yourself, execute the various wrappers ([`runscript_nonperiodic.sh`](runscript_nonperiodic.sh), [`runscript_nonperiodic_doubleprec.sh`](runscript_nonperiodic_doubleprec.sh), [`runscript_periodic.sh`](runscript_periodic.sh), [`runscript_periodic_doubleprec.sh`](runscript_periodic_doubleprec.sh)), which internally call the [`benchmark_cost_vs_accuracy.py`](benchmark_cost_vs_accuracy.py) script with the correct command-line arguments.
To afterwards combine the results from single- and double-precision runs into a single informative plot, the [`plot_combined.ipynb`](plot_combined.ipynb) notebook is provided.
