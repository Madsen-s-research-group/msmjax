import jax
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


@jax.custom_jvp
def _sqrt(x):
    # see https://gitlab.tuwien.ac.at/e165-03-1_theoretische_materialchemie/scripts-et-al/-/wikis/Sqrt-without-trivial-NaN-derivative-for-jax
    return jnp.sqrt(x)


@_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    (x,) = primals
    (xdot,) = tangents
    primal_out = _sqrt(x)
    tangent_out = jnp.where(x == 0.0, 0.0, 0.5 / primal_out) * xdot
    return (primal_out, tangent_out)
