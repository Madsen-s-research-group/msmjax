# Copyright 2025 The msmJAX contributors
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTE: This file has been modified by the msmJAX contributors by removing
# all module members except _diagonal_mask which was adapted to work for
# non-square matrices and renamed _generalized_diagonal_mask.
# Additionally, it was modified by removing no-longer-used imports,
# by making changes for compatibility with recent JAX versions, and by
# reformatting for consistency with msmJAX's code style.

"""Code to transform functions on individual tuples of particles to sets."""

import jax.numpy as jnp

from .util import Array


def _generalized_diagonal_mask(a: Array) -> Array:
    """Set the diagonal of a, possibly wider than tall, matrix to zero.

    Adapted from the JAX-MD function `jax_md.smap._diagonal_mask`.
    Compared to the JAX-MD version, it only handles two-dimensional matrices,
    but they may be wider than tall.

    .. warning::
       Any NaN or infinite entries (including ones off the diagonal!) will
       be silently replaced by this function. For the diagonal, this is
       usually reasonable and desired. That is because in the case for which
       this function is designed, diagonal elements of the input correspond
       to interactions of particles with themselves, which are usually
       considered artifactual and which may be undefined. When off-diagonal
       elements are replaced this way, however, this may obscure the origin
       of bugs that caused them to be invalid.

    Args:
        a: Original matrix.

    Returns:
        The matrix with diagonal set to zero.
    """
    if len(a.shape) != 2:
        raise ValueError("Only two-dimensional arrays are supported.")
    M, N = a.shape
    if M > N:
        raise ValueError(
            "Input array must be either square, or wider than tall."
        )
    a = jnp.nan_to_num(a)
    mask = 1.0 - jnp.eye(M, dtype=a.dtype)
    mask = jnp.pad(
        mask,
        pad_width=((0, 0), (0, N - M)),
        mode="constant",
        constant_values=1,
    )
    return mask * a
