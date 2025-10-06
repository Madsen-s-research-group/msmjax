# msmjax

**msmJAX** is an implementation of the multilevel summation method (MSM) with B-spline interpolation for the evaluation of electrostatic interactions in Python, built on Google JAX.
The implementation is mostly based on Ref. 1, with additional inputs from Refs. 2 and 3.

This page will be updated with links to ensuing publications.

## Getting started
Tutorials and documentation can be found on the [GitHub pages](https://madsen-s-research-group.github.io/msmJAX/).
More elaborate examples and benchmarks can be found in the `examples/` directory.

## Installation
We recommend working in a virtual environment to isolate your installation, e.g., using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [venv](https://docs.python.org/3/library/venv.html).

### CPU-only installation
Either clone the msmJAX repository, navigate to the repository root, and run: 
```bash
    pip install -U pip
    pip install -e .
```

Or, install from pypi:
```bash
    pip install -U pip
    pip install msmjax
```

### CUDA installation
To get the most out of msmJAX, you will want GPU support with CUDA.
To that end, install JAX with CUDA before installing msmJAX.

The simplest way to get a CUDA-enabled JAX installation is to install CUDA and cuDNN from pip wheels as demonstrated in the below snippet:
```bash
    pip install -U pip
    pip install -U "jax[cuda12]"
    pip install -e .
```

To use a preinstalled local CUDA, see the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
Generally speaking, the JAX ecosystem is fast-moving, and checking the current recommended installation method might be a good idea.

### Optional dependencies

To run the tests, proceed like above, but install msmJAX with `pip install -e ".[test]"` or `pip install msmjax[test]`.
Then, you can run `pytest tests/`, which is recommended to verify that the installation was successful.

To run the examples, install with `pip install -e ".[examples]"` or `pip install msmjax[examples]`.
In some of the examples, additional external programs are used.
But these are not required for a basic run of the examples, only if you want to re-perform the setting up the inputs or certain post-processing tasks yourself.
Where they are used, this is indicated in the description of the individual examples.

To build the documentation yourself, install with `pip install -e ".[doc]"` or `pip install msmjax[doc]`, then navigate to `docs/` and run `make html`.

## License

msmJAX is distributed with the [Apache-2.0 license](https://github.com/Madsen-s-research-group/msmJAX/blob/public_release_v2025.08.1/LICENSE).

It also contains code derived from the [JAX-MD](https://github.com/jax-md/jax-md) third-party package ([release v0.2.24](https://github.com/jax-md/jax-md/releases/tag/jax-md-v0.2.24)), in the [jax_md](https://github.com/Madsen-s-research-group/msmJAX/tree/public_release_v2025.08.1/src/msmjax/jax_md) directory.

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
