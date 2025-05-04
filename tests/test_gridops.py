"""Tests for grid operations specific to B-spline interpolation.

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from msmjax.bspline.gridops import (
    _arbitrary_dim_outer,
    _make_prolongate_1d,
    _make_restrict_1d,
    _multiindex_outer,
    _ravel_multi_index_with_invalidation,
    make_basis_evaluation_fn,
    make_prolongation_operator,
    make_restriction_operator,
)
