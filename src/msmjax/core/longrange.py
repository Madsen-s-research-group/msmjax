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

from typing import Any, Callable, Literal, Optional, ParamSpec, Sequence

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
    # TODO: function name?
    pass


def make_anterpolation_fn():
    pass


P = ParamSpec("P")


def make_compute_longrange(
    anterpolation_fn: Callable[[ArrayLike, ArrayLike, P], Array],
    grid_pass_fn: Callable[[ArrayLike, P], Array],
    interpolation_fn: Callable[[ArrayLike, ArrayLike, ArrayLike, P], Array],
) -> Callable[[ArrayLike, ArrayLike, P], Any]:
    def compute_longrange(
        positions: ArrayLike, charges: ArrayLike, **kwargs: P.kwargs
    ) -> Any:
        # TODO: cell, kwargs?
        gridcharge_lvl_one = anterpolation_fn(positions, charges, **kwargs)
        gridpotential_lvl_one = grid_pass_fn(gridcharge_lvl_one, **kwargs)
        result = interpolation_fn(
            gridpotential_lvl_one, positions, charges, **kwargs
        )
        return result

    return compute_longrange


def make_energy_interpolation_fn(basis_eval_fn):
    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ):
        basis_vals, indices = basis_eval_fn(positions)
        # TODO: Do we need to use a fill value with `take` here?
        #  (it shouldn't be possible for indices returned by the spline eval
        #  functions to be out of bounds)
        particle_contribs = charges * (
            gridpotential.take(indices) * basis_vals
        ).sum(axis=1)
        energy = 0.5 * jnp.sum(particle_contribs)
        return energy

    return compute
