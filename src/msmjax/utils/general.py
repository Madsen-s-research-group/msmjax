from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax import numpy as jnp
from jax._src.basearray import ArrayLike

# Type definitions
CellMode = Literal["ortho", "triclinic"]
ConvMeth = Literal["scipy-direct", "scipy-fft"]
KernelFn = Callable[[ArrayLike], Array]


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
    # TODO: This reference to our gitlab should not be left in
    # see https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/scripts-et-al/-/wikis/Sqrt-without-trivial-NaN-derivative-for-jax
    return jnp.sqrt(x)


@_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    (x,) = primals
    (xdot,) = tangents
    primal_out = _sqrt(x)
    tangent_out = jnp.where(x == 0.0, 0.0, 0.5 / primal_out) * xdot
    return (primal_out, tangent_out)


def get_max_cutoff_3d(cell: jnp.ndarray):
    """Get the maximum cutoff value that fits into a 3D cell.

    Args:
        cell: Cell, shape=(3, 3).

    Returns:
        Cutoff radius
    """
    # TODO: Move to core.shortrange or leave in utils?
    return jnp.min(
        jnp.fabs(
            jnp.linalg.det(cell)
            / jnp.array(
                [
                    jnp.linalg.norm(jnp.cross(i, j))
                    for i, j in zip(cell, jnp.roll(cell, 1, axis=0))
                ]
            )
        )
        / 2.0
    )


def get_max_cutoff_for_mic(cell: ArrayLike):
    """Get the maximum cutoff value that fits into a given cell.

    I.e., the maximum cutoff inside which distances calculated using the
    minimum-image convention (MIC) are guaranteed to be calculated correctly.

    Args:
        cell: Array representing unit cell, shape `(n_dim, n_dim)`.

    Raises:
        ValueError: If `cell` has invalid spatiol dimension.

    Returns:
        Cutoff radius
    """
    # TODO: Move to core.shortrange or leave in utils?
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
