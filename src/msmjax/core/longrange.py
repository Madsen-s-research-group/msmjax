"""Generic code for long-range part (that is evaluated using grids).

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

from typing import Callable, Literal, Optional, ParamSpec, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

# TODO: This should be defined elsewhere, since the shortrange part also uses it
CellMode = Literal["ortho", "general"]
KernelFn = Callable[[ArrayLike], Array]  # TODO: float or Array?


def make_grid_pass(
    restriction_fns: Sequence[Callable],
    prolongation_fns: Sequence[Callable],
    interaction_fns: Sequence[Callable],  # TODO: name
) -> Callable[[ArrayLike], Array]:
    pass


def make_anterpolation_fn():
    pass


def make_full_pass():
    pass
