"""Code for short-range part (that is directly evaluated, without grids).

    References:
        [1] Hardy, D. J.; Wolff, M. A.; Xia, J.; Schulten, K.; Skeel,
        R. D. Multilevel Summation with B-Spline Interpolation for Pairwise
        Interactions in Molecular Dynamics Simulations. J. Chem. Phys. 2016,
        144 (11), 114112. https://doi.org/10.1063/1.4943868.

        [2] Hardy, D. J. Multilevel Summation for the Fast Evaluation of
        Forces for the Simulation of Biomolecules (PhD thesis), University
        of Illinois at Urbana-Champaign, 2006.
"""

from typing import Callable, Tuple

from jax_md import partition, space


def make_evaluate_shortrange_with_neighbor_list(
    shortrange_kernel: Callable,
    cutoff: float,  # TODO
    displacement_or_metric,
    box,
    **neighbor_kwargs,
) -> Tuple[partition.NeighborFn, Callable]:
    neighbor_fn = partition.neighbor_list(
        displacement_or_metric=displacement_or_metric,
        box=box,
        r_cutoff=cutoff,
        **neighbor_kwargs,
    )

    return neighbor_fn
