"""General utilities"""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as onp
from jax import Array
from jax import numpy as jnp
from jax._src.basearray import ArrayLike

# Type definitions
CellMode = Literal["ortho", "triclinic"]
ConvMeth = Literal["scipy-direct", "scipy-fft"]
KernelFn = Callable[[ArrayLike], Array]

inds_matrix_to_six_component_stress = (
    jnp.array([0, 1, 2, 0, 0, 1]),
    jnp.array([0, 1, 2, 1, 2, 2]),
)


def _divide_zero_safe(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
) -> jnp.ndarray:
    """Function that forces the result of dividing by 0 to be equal to 0.0
    in a jit- and autodiff-compatible way

    Args:
        numerator: Values in the numerator
        denominator: Values in the denominator, may contain zeros
    Returns:
        numerator / denominator with result == 0.0 where denominator == 0.0
    """
    denominator_masked = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(
        denominator == 0.0,
        0.0,
        numerator / denominator_masked,
    )


@jax.custom_jvp
def _sqrt(x):
    return jnp.sqrt(x)


@_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    (x,) = primals
    (xdot,) = tangents
    primal_out = _sqrt(x)
    tangent_out = jnp.where(x == 0.0, 0.0, 0.5 / primal_out) * xdot
    return (primal_out, tangent_out)


def get_max_cutoff_for_mic(cell: ArrayLike):
    """Get the maximum cutoff value that fits into a given cell.

    I.e., the maximum cutoff inside which distances calculated using the
    minimum-image convention (MIC) are guaranteed to be calculated correctly.

    Args:
        cell: Array representing unit cell, shape `(n_dim, n_dim)`.

    Raises:
        ValueError: If `cell` has invalid spatial dimension.

    Returns:
        Cutoff radius
    """
    n_dim = cell.shape[0]

    if n_dim == 1:
        length = cell[0, 0]
        return 0.5 * length
    elif n_dim == 2:
        cell_area = jnp.linalg.norm(jnp.cross(cell[0], cell[1]))
        side_lengths = jnp.linalg.norm(cell, axis=1)
        return 0.5 * cell_area / jnp.max(side_lengths)
    elif n_dim == 3:
        cell_volume = jnp.abs(jnp.linalg.det(cell))
        face_areas = jnp.array(
            [
                jnp.linalg.norm(jnp.cross(cell[i], cell[j]))
                for i, j in zip(*jnp.triu_indices(n_dim, k=1))
            ]
        )
        return 0.5 * cell_volume / jnp.max(face_areas)
    else:
        raise ValueError("Number of dimensions must be 1, 2 or 3.")


def find_covering_grid_extents(
    grid_axes: ArrayLike, spacings: ArrayLike, cutoff: float
):
    n_dim = grid_axes.shape[0]
    inverse = onp.linalg.inv(grid_axes)

    if n_dim == 1:
        return tuple(onp.atleast_1d(cutoff / spacings).astype(int).tolist())
    elif n_dim == 2:
        phis = onp.linspace(0, 2 * onp.pi, 500)
        points_unitsphere = onp.array([onp.cos(phis), onp.sin(phis)]).T
    elif n_dim == 3:
        phis = onp.linspace(0, 2 * onp.pi, 200)
        thetas = onp.linspace(0, onp.pi, 200)
        phis, thetas = onp.meshgrid(phis, thetas)
        phis = phis.ravel()
        thetas = thetas.ravel()
        points_unitsphere = onp.array(
            [
                onp.cos(phis) * onp.sin(thetas),
                onp.sin(phis) * onp.sin(thetas),
                onp.cos(thetas),
            ]
        ).T
    else:
        raise ValueError("Spatial dimensions greater than 3 not supported.")

    points_at_cutoff = cutoff * points_unitsphere
    points_at_cutoff_transformed = points_at_cutoff @ inverse

    single_grid_cell = (
        grid_axes
        / onp.linalg.norm(grid_axes, axis=1)[:, onp.newaxis]
        * onp.atleast_1d(spacings)[:, onp.newaxis]
    )
    # In the transformed system, grid cells are represented by diagonal
    # matrices:
    spacings_transformed = onp.diag(single_grid_cell @ inverse)

    # The maximum taken from the precomputed cutoff sphere points may be
    # slightly too low, because they incompletely sample the cutoff sphere
    # => use an additional small tolerance
    tol = 1.0e-3
    sizes_from_center = onp.floor(
        points_at_cutoff_transformed.max(axis=0) / spacings_transformed + tol
    ).astype(int)

    return tuple(sizes_from_center.tolist())
