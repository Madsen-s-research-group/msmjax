import jax.numpy as jnp


# TODO: centralize this function in one place (originally taken from bspline_basis)
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
