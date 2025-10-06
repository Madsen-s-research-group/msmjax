Installation
==================================
We recommend working in a virtual environment to isolate your installation, e.g., using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [venv](https://docs.python.org/3/library/venv.html).

CPU-only installation
----------------------------------
Either clone the msmJAX repository, navigate to the repository root, and run::

    pip install -U pip
    pip install -e .

Or, install from pypi::

    pip install -U pip
    pip install msmjax

CUDA installation
----------------------------------
To get the most out of msmJAX, you will want GPU support with CUDA.
To that end, install JAX with CUDA before installing msmJAX.

The simplest way to get a CUDA-enabled JAX installation is to install CUDA and cuDNN from pip wheels as demonstrated in the below snippet::

    pip install -U pip
    pip install -U "jax[cuda12]"
    pip install -e .

To use a preinstalled local CUDA, see the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
Generally speaking, the JAX ecosystem is fast-moving, and checking the current recommended installation method might be a good idea.

Optional dependencies
----------------------------------

To run the tests, proceed like above, but install msmJAX with `pip install -e ".[test]"` or `pip install msmjax[test]`.
Then, you can run `pytest tests/`, which is recommended to verify that the installation was successful.

To run the examples, install with `pip install -e ".[examples]"` or `pip install msmjax[examples]`.
In some of the examples, additional external programs are used.
But these are not required for a basic run of the examples, only if you want to re-perform the setting up the inputs or certain post-processing tasks yourself.
Where they are used, this is indicated in the description of the individual examples.

To build the documentation yourself, install with `pip install -e ".[doc]"` or `pip install msmjax[doc]`, then navigate to `docs/` and run `make html`.
