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
# code not needed within msmJAX and making corresponding changes to imports,
# by making changes for compatibility with recent JAX versions, and by
# reformatting for consistency with msmJAX's code style.

"""Spaces in which particles are simulated.

Spaces are pairs of functions containing:
  `displacement_fn(Ra, Rb, **kwargs)`:
    Computes displacements between pairs of particles. `Ra` and `Rb` should
    be ndarrays of shape `[spatial_dim]`. Returns an ndarray of shape `[spatial_dim]`.
    To compute the displacement over more than one particle at a time see the
    :meth:`map_product`, :meth:`map_bond`, and :meth:`map_neighbor` functions.
  `shift_fn(R, dR, **kwargs)`:
    Moves points at position `R` by an amount `dR`.

Spaces can accept keyword arguments allowing the space to be changed over the
course of a simulation. For an example of this use see :meth:`periodic_general`.

Although displacement functions are compute the displacement between two
points, it is often useful to compute displacements between multiple particles
in a vectorized fashion. To do this we provide three functions: `map_product`,
`map_bond`, and `map_neighbor`:
  map_product:
    Computes displacements between all pairs of points such that if
    `Ra` has shape `[n, spatial_dim]` and `Rb` has shape `[m, spatial_dim]` then the
    output has shape `[n, m, spatial_dim]`.
  map_bond:
    Computes displacements between all points in a list such that if
    `Ra` has shape `[n, spatial_dim]` and `Rb` has shape `[m, spatial_dim]` then the
    output has shape `[n, spatial_dim]`.
  map_neighbor:
    Computes displacements between points and all of their
    neighbors such that if `Ra` has shape `[n, spatial_dim]` and `Rb` has shape
    `[n, neighbors, spatial_dim]` then the output has shape
    `[n, neighbors, spatial_dim]`.
"""

from typing import Callable, Tuple, Union

import jax.numpy as jnp
from jax import vmap

from .util import Array, safe_mask

# Types


DisplacementFn = Callable[[Array, Array], Array]
MetricFn = Callable[[Array, Array], float]
DisplacementOrMetricFn = Union[DisplacementFn, MetricFn]

ShiftFn = Callable[[Array, Array], Array]

Space = Tuple[DisplacementFn, ShiftFn]
Box = Array


def square_distance(dR: Array) -> Array:
    """Computes square distances.

    Args:
      dR: Matrix of displacements; `ndarray(shape=[..., spatial_dim])`.
    Returns:
      Matrix of squared distances; `ndarray(shape=[...])`.
    """
    return jnp.sum(dR**2, axis=-1)


def distance(dR: Array) -> Array:
    """Computes distances.

    Args:
      dR: Matrix of displacements; `ndarray(shape=[..., spatial_dim])`.
    Returns:
      Matrix of distances; `ndarray(shape=[...])`.
    """
    dr = square_distance(dR)
    return safe_mask(dr > 0, jnp.sqrt, dr)


def metric(displacement: DisplacementFn) -> MetricFn:
    """Takes a displacement function and creates a metric."""
    return lambda Ra, Rb, **kwargs: distance(displacement(Ra, Rb, **kwargs))


def map_product(
    metric_or_displacement: DisplacementOrMetricFn,
) -> DisplacementOrMetricFn:
    """Vectorizes a metric or displacement function over all pairs."""
    return vmap(vmap(metric_or_displacement, (0, None), 0), (None, 0), 0)


def map_bond(
    metric_or_displacement: DisplacementOrMetricFn,
) -> DisplacementOrMetricFn:
    """Vectorizes a metric or displacement function over bonds."""
    return vmap(metric_or_displacement, (0, 0), 0)
