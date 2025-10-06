Demonstration of performance gain from custom derivative implementations for the long-range energy contribution $U^{1+}$.

Example results, obtained on an NVIDIA A40 GPU, using jax and jaxlib version 0.7, and CUDA toolkit version 12.9, are contained in `example_results/`.

To try the benchmarks yourself, execute the wrapper scripts [`runscript.sh`](runscript.sh) and [`runscript_doubleprec.sh`](runscript_doubleprec.sh) for running in single and double precision, respectively.
They internally call the [`benchmark_custom_longrange_derivative.py`](benchmark_custom_longrange_derivative.py) script with appropriate parameters.
