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

"""Defines utility functions."""

from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jit

Array = jnp.ndarray
PyTree = Any

i16 = jnp.int16
i32 = jnp.int32
i64 = jnp.int64

f32 = jnp.float32
f64 = jnp.float64


@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)
