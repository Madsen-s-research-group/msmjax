from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as onp

from msmjax.bspline_interpolation.basis import create_bspline_basis_element


def _arbitrary_dim_outer(*xi: jax.Array) -> jax.Array:
    """Compute the outer product of an arbitrary number of arrays"""
    # TODO: could this be made more efficient? (the way it is currently done
    #   technically involves redundant multiplications)
    return jnp.prod(jnp.array(jnp.meshgrid(*xi, indexing="ij")), axis=0)


def _multiindex_outer(*inds_individual_axes):
    # TODO: name of this function and its arguments?
    multi_inds = tuple(
        arr.ravel()
        for arr in jnp.meshgrid(*inds_individual_axes, indexing="ij")
    )
    return multi_inds


# TODO: jit with static pbc?
# TODO: function name?
def _ravel_multi_index_with_invalidation(multi_index, dims, pbc):
    pbc = onp.asarray(pbc)  # TODO: onp/jnp?
    all_periodic = pbc.all()
    any_periodic = pbc.any()
    intentionally_out_of_bounds_index = onp.prod(dims)

    flat_inds_wrapped = jnp.ravel_multi_index(multi_index, dims, mode="wrap")
    if all_periodic:
        return flat_inds_wrapped
    else:
        multi_index = jnp.asarray(multi_index)
        in_bounds = jnp.logical_and(
            multi_index >= 0, multi_index < jnp.array(dims)[:, jnp.newaxis]
        )
        if any_periodic:
            in_bounds = jnp.logical_or(in_bounds, pbc[:, jnp.newaxis])
        return jnp.where(
            # TODO: Which axis? Shouldn't it be .all() instead of .any()?
            in_bounds.all(axis=0),
            flat_inds_wrapped,
            intentionally_out_of_bounds_index,
        )


def make_basis_evaluation_fn(
    grid_shape: tuple[int, ...], p: int, pbc: Sequence[bool]
):
    pbc = onp.asarray(pbc)

    # TODO: External factory function that creates both `zero_align_idx` and
    #  `to_positional_idx` from `periodic` and `p`

    def zero_align_idx(positional_idx):
        if pbc.all():
            return positional_idx
        elif (~pbc).all():
            return positional_idx - p // 2
        else:
            raise ValueError  # TODO: mixed BCs

    def to_positional_idx(zero_aligned_idx):
        if pbc.all():
            return zero_aligned_idx
        elif (~pbc).all():
            return zero_aligned_idx + p // 2
        else:
            raise ValueError  # TODO: mixed BCs

    bspline_basis_element = create_bspline_basis_element(order=p - 1)

    def eval_basis(
        coords: jax.Array, spacings: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # TODO: should spacing be argument to the closure or to the setup fn?
        r_over_h = coords / spacings
        raw_reference_inds = jnp.ceil(r_over_h).astype(int)
        raw_inds = raw_reference_inds[:, jnp.newaxis] + jnp.arange(
            -p // 2, p // 2
        )
        splinevals_per_axis = jax.vmap(jax.vmap(bspline_basis_element))(
            (r_over_h - raw_inds.T).T
        )
        inds_per_axis = to_positional_idx(raw_inds)

        splinevals = _arbitrary_dim_outer(*splinevals_per_axis).ravel()

        multiinds = _multiindex_outer(*inds_per_axis)
        flat_inds = _ravel_multi_index_with_invalidation(
            multiinds, dims=grid_shape, pbc=pbc
        )

        return splinevals, flat_inds

    return eval_basis
