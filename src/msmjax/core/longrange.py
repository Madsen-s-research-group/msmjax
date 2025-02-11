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

from functools import partial
from typing import Callable, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax.typing import ArrayLike


@partial(jax.jit, static_argnames=["pbc", "method"])
def convolve_scipy_general_pbc(
    data: ArrayLike,
    kernel: ArrayLike,
    pbc: Sequence[bool],
    method: Literal["direct", "fft"],  # TODO: centralize definition?
) -> Array:
    # TODO: Make this a protected function (_convolve_scipy_general_pbc)?

    pbc = onp.asarray(pbc)

    if pbc.any():
        size_kernel = onp.array(kernel.shape)
        size_kernel_below_middle = size_kernel // 2
        size_kernel_above_middle = size_kernel - size_kernel_below_middle - 1
        pad_width = tuple(
            (int(s_b), int(s_a))
            for s_b, s_a in zip(
                size_kernel_below_middle, size_kernel_above_middle
            )
        )
        pad_width = tuple(
            pw if periodic else (0, 0) for pw, periodic in zip(pad_width, pbc)
        )
        inds_reconstruct_unpadded = []
        for pw, periodic in zip(pad_width, pbc):
            if periodic:
                inds_reconstruct_unpadded.append(slice(pw[0], -pw[1]))
            else:
                inds_reconstruct_unpadded.append(slice(None))
        inds_reconstruct_unpadded = tuple(inds_reconstruct_unpadded)
        data_extended = jnp.pad(
            data,
            pad_width=pad_width,
            mode="wrap",
        )
        nruter = jax.scipy.signal.convolve(
            data_extended, kernel, mode="same", method=method
        )
        return nruter[inds_reconstruct_unpadded]
    else:
        return jax.scipy.signal.convolve(
            data, kernel, mode="same", method=method
        )


def make_anterpolation_fn(basis_eval_fn, grid_shape: tuple[int, ...]):
    grid_size = int(onp.prod(grid_shape))

    def anterpolate(positions: ArrayLike, charges: ArrayLike) -> Array:
        """Anterpolate charge from particles to grid"""
        # TODO: Where to take size, shape from? (Are they even really needed?)
        # TODO: flat vs. multidim? It seems that `indices` is expected to be
        #  flat, but should this really be the case universally?
        basis_vals, indices = basis_eval_fn(positions)
        gridcharge_flat = jnp.zeros(grid_size)
        gridcharge_flat = gridcharge_flat.at[indices].add(
            charges[:, jnp.newaxis] * basis_vals
        )
        return gridcharge_flat.reshape(grid_shape)

    return anterpolate


def _interpolate_potential():
    # TODO: Do we want to provide this?
    pass


def _interpolate_energy(
    gridpotential: ArrayLike,
    basis_vals: ArrayLike,
    indices: ArrayLike,
    charges: ArrayLike,
):
    # TODO: Do we need to use a fill value with `take` here?
    #  (it shouldn't be possible for indices returned by the spline eval
    #  functions to be out of bounds)
    energy = 0.5 * jnp.sum(
        charges * (gridpotential.take(indices) * basis_vals).sum(axis=1)
    )
    return energy


def _interpolate_forces(
    gridpotential: ArrayLike,
    basis_grads: ArrayLike,
    indices: ArrayLike,
    charges: ArrayLike,
):
    # TODO: Do we need to use a fill value with `take` here?
    #  (it shouldn't be possible for indices returned by the spline eval
    #  functions to be out of bounds)
    forces = -charges[:, jnp.newaxis] * jnp.sum(
        gridpotential.take(indices)[..., jnp.newaxis] * basis_grads, axis=1
    )
    return forces


def make_energy_interpolation_fn(basis_eval_fn):
    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ):
        basis_vals, indices = basis_eval_fn(positions)
        return _interpolate_energy(gridpotential, basis_vals, indices, charges)

    return compute


def make_forces_interpolation_fn(basis_grad_fn):
    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ):
        basis_grads, indices = basis_grad_fn(positions)
        return _interpolate_forces(
            gridpotential, basis_grads, indices, charges
        )

    return compute


def make_energy_and_forces_interpolation_fn(basis_val_and_grad_fn):
    def compute(
        gridpotential: ArrayLike, positions: ArrayLike, charges: ArrayLike
    ):
        # TODO: Make sure that the expected structure of the return value of
        #  basis_val_and_grad_fn is properly documented, and correctly used
        #  elsewhere.
        (basis_vals, indices), basis_grads = basis_val_and_grad_fn(positions)
        energy = _interpolate_energy(
            gridpotential, basis_vals, indices, charges
        )
        forces = _interpolate_forces(
            gridpotential, basis_grads, indices, charges
        )
        return energy, forces

    return compute


def make_unitcube_transform_fns_ortho(
    cell: ArrayLike,
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array]]:
    # TODO: Right module for this function?
    inverse = 1.0 / jnp.diag(cell)
    transform_pos = lambda x: x * inverse
    backtransform_grad = lambda dx: dx * inverse
    return transform_pos, backtransform_grad


def make_unitcube_transform_fns_general(
    cell: ArrayLike,
) -> tuple[Callable[[ArrayLike], Array], Callable[[ArrayLike], Array]]:
    # TODO: Right module for this function?
    inverse = jnp.linalg.pinv(cell)
    transform_pos = lambda x: x @ inverse
    backtransform_grad = lambda dx: dx @ inverse.T
    return transform_pos, backtransform_grad


def make_grid_pass(
    restriction_fns: Sequence[Callable],
    prolongation_fns: Sequence[Callable],
    interaction_fns: Sequence[Callable],  # TODO: name (everywhere)
) -> Callable[[ArrayLike, Sequence[ArrayLike]], Array]:
    # TODO: In fact it's questionable, whether a separate interaction_fn for
    #  each level is needed at all. `convolve_scipy_general_pbc` should work
    #  for all levels, shouldn't it?
    #  Do we still want to offer separate functions for increased flexibility?
    #  Even for restriction and prolongation a single function might be sufficient?
    if (
        not len(restriction_fns)
        == len(prolongation_fns)
        == len(interaction_fns)
    ):
        raise ValueError(
            "restriction_fns, prolongation_fns, interaction_fns "
            "must all have same length."
        )
    n_levels = len(restriction_fns) - 1

    # TODO: Name of the returned function?
    def grid_pass(
        gridcharge_lvl_one: ArrayLike, kernel_stencils: Sequence[ArrayLike]
    ) -> Array:
        if not len(kernel_stencils) == n_levels + 1:
            raise ValueError(
                "Wrong number of kernel stencils. "
                "Expected {} (including a placeholder at level zero), "
                "got {}.".format(n_levels + 1, len(kernel_stencils))
            )
        # TODO: list instead of dict?
        gridcharges_all_levels = {1: gridcharge_lvl_one}

        # TODO: More efficient to do the pass in the following order instead?
        #  1. Iteratively go up (compute grid charges at all levels)
        #  2. Apply interactions at all levels (possibly via `tree_map`)
        #  3. Iteratively go back down.

        # Go up ladder
        for lvl in range(2, n_levels + 1):
            restrict = restriction_fns[lvl]
            gridcharge_fine = gridcharges_all_levels[lvl - 1]
            gridcharge_coarse = restrict(gridcharge_fine)
            gridcharges_all_levels[lvl] = gridcharge_coarse

        # Apply top-level interaction
        gridcharge_toplevel = gridcharges_all_levels[n_levels]
        interact_toplevel = interaction_fns[n_levels]
        kernel_stencils_toplevel = kernel_stencils[n_levels]
        gridpotential = interact_toplevel(
            gridcharge_toplevel, kernel_stencils_toplevel
        )

        # Go down ladder
        for lvl in range(n_levels - 1, 0, -1):
            gridpotential = interaction_fns[lvl](
                gridcharges_all_levels[lvl], kernel_stencils[lvl]
            ) + prolongation_fns[lvl](gridpotential)

        return gridpotential

    return grid_pass


# TODO: Also add function for the calculation of energy by direct
#  contraction of grid charges with grid potential (without going the route
#  of reconstructing electrostatic potential by interpolation)


def make_compute_longrange(
    anterpolation_fn: Callable[[ArrayLike, ArrayLike], Array],
    grid_pass_fn: Callable[[ArrayLike, Optional[Sequence[ArrayLike]]], Array],
    interpolation_fn: Callable[[ArrayLike, ArrayLike, ArrayLike], Array],
    kernel_stencil_construction_fn: Callable[[...], Sequence[Array]] = None,
) -> Callable[[ArrayLike, ArrayLike, ...], Array]:
    def compute_longrange(
        positions: ArrayLike, charges: ArrayLike, **kwargs: ...
    ) -> Array:
        # TODO: Should this rather be a lower-level function
        #  (`_make_compute_longrange`?, other name?) that takes
        #  positions, charges, kernel_stencils?
        #  (In which case: `static` in name?)
        gridcharge_lvl_one = anterpolation_fn(positions, charges)
        if kernel_stencil_construction_fn is None:
            gridpotential_lvl_one = grid_pass_fn(gridcharge_lvl_one)
        else:
            gridpotential_lvl_one = grid_pass_fn(
                gridcharge_lvl_one, kernel_stencil_construction_fn(**kwargs)
            )
        result = interpolation_fn(gridpotential_lvl_one, positions, charges)

        return result

    return compute_longrange


def make_compute_u_oneplus(
    basis_eval_fn,  # TODO: `unitcube` in name? `lvl_one` in name?
    grid_pass_fn,  # TODO: `unitcube` in name?
    kernel_stencil_construction_fn,
    cell_mode,
):
    # TODO: dynamic_cell (or similar) in name? (only if that's what the
    #  function actually ends up doing, though (see below))
    # TODO: Mutually exclusive arguments `kernel_stencils` (indicating
    #  static-cell version) and `kernel_stencil_construction_fn` (indicating
    #  dynamic-cell version)?
    # TODO: `cell_mode=None` for no unitcube transform?

    if cell_mode == "ortho":
        make_unitcube_transform_fns = make_unitcube_transform_fns_ortho
    elif cell_mode == "general":
        make_unitcube_transform_fns = make_unitcube_transform_fns_general
    else:
        raise ValueError("Invalid `cell_mode`.")

    anterpolation_fn = make_anterpolation_fn(basis_eval_fn, grid_shape)
    interpolation_fn = make_energy_interpolation_fn(basis_eval_fn)
    compute_unitcube = make_compute_longrange(
        anterpolation_fn=anterpolation_fn,
        grid_pass_fn=grid_pass_fn,
        interpolation_fn=interpolation_fn,
    )

    def compute(positions, charges, cell):
        transform_pos, backtransform_grad = make_unitcube_transform_fns(cell)
        return compute_unitcube(
            transform_pos(positions),
            charges,
            kernel_stencil_construction_fn(cell),
        )

    return compute
