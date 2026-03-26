Comparison of the scaling of run time as a function of particle number between different algorithms:
msmJAX is compared with the Ewald and PME implementations from JAX-MD v0.2.25.

Example results, obtained on an NVIDIA A40 GPU, using jax and jaxlib version 0.9.1, and CUDA toolkit version 12.9, are contained in `example_results/`.

To run the benchmark yourself, execute the various wrappers (`runscript_msm.sh`, `runscript_ewald.sh`, `runscript_pme.sh`), which internally call the `time_algo.py` script with appropriate command-line arguments.
For plotting the results after running, the `plot_comparison.ipynb` notebook can be used.

The sets of algorithm parameters for which timing experiments are run are stored in `prepared_data/`.
In the case of Ewald and PME they were first pre-screened for viability by means of error estimation formulas from the literature (Petersen, Henrik G. “Accuracy and Efficiency of the Particle Mesh Ewald Method.” The Journal of Chemical Physics 103, no. 9 (1995): 3668–79. https://doi.org/10.1063/1.470043.).
