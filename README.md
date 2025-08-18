# msmjax

**msmJAX** is an implementation of the multilevel summation method (MSM) with B-spline interpolation for the evaluation of electrostatic interactions in Python, built on Google JAX.
The implementation is mostly based on Ref. 1, with additional inputs from Refs. 2 and 3.

This page will be updated with links to ensuing publications.

## Getting started

[//]: # (TODO)
**TODO: Correct link to github pages**

[//]: # (TODO)
Tutorials and documentation can be found on the [GitHub pages](TODO).
More elaborate examples and benchmarks can be found in the `examples/` directory.

## Installation
First clone the msmJAX repository and navigate to the repository root.

We recommend working in a virtual environment to isolate your installation, e.g., using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [venv](https://docs.python.org/3/library/venv.html).

### CPU-only installation
From the msmJAX repository root, run 
```bash
    pip install -U pip
    pip install -e .
```

### CUDA installation
To get the most out of msmJAX, you will want GPU support with CUDA.
To that end, install JAX with CUDA *before* installing msmJAX.

The simplest way to get a CUDA-enabled JAX installation is to install CUDA and cuDNN from pip wheels as demonstrated in the below snippet:
```bash
    pip install -U pip
    pip install -U "jax[cuda12]"
    pip install -e .
```

To use a preinstalled local CUDA, see the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
Generally speaking, the JAX ecosystem is fast-moving, and checking the current recommended installation method might be a good idea.

### Optional dependencies

To run the tests, proceed like above, but install msmJAX with `pip install -e ".[test]"`.
Then, you can run `pytest tests/`, which is recommended to verify that the installation was successful.

To run the examples, install with `pip install -e ".[examples]"`
In some of the examples, additional external programs are used.
But these are not required for a basic run of the examples, only if you want to re-perform the setting up the inputs or certain post-processing tasks yourself.
Where they are used, this is indicated in the description of the individual examples.

## References

1. D. J. Hardy, M. A. Wolff, J. Xia, K. Schulten, and R. D. Skeel, <br>
    “Multilevel summation with B-spline interpolation for pairwise interactions in molecular dynamics simulations”, <br>
    J. Chem. Phys., vol. 144, no. 11, p. 114112, Mar. 2016, doi: [10.1063/1.4943868](https://doi.org/10.1063/1.4943868).
2. D. J. Hardy, Z. Wu, J. C. Phillips, J. E. Stone, R. D. Skeel, and K. Schulten, <br>
    “Multilevel Summation Method for Electrostatic Force Evaluation”, <br>
    J. Chem. Theory Comput., vol. 11, no. 2, pp. 766–779, Feb. 2015, doi: [10.1021/ct5009075](https://doi.org/10.1021/ct5009075).
3. D. J. Hardy,<br>
    “Multilevel Summation for the Fast Evaluation of Forces for the Simulation of Biomolecules,”<br>
    University of Illinois at Urbana-Champaign, 2006.
