"""Benchmark data for JAX users."""

import jax
import jax.numpy as jnp

from src.data.dataset import load_samples


def load_jax_samples(split):
    """Samples of "train" or "validation" as JAX arrays.

    inputs: (n_samples, 3), the 2 parameters followed by the time
    fields: (n_samples, 5454), the pressure on the 2727 cells followed by the saturation
            on the same cells
    """
    inputs, fields = load_samples(split)
    return jnp.asarray(inputs), jnp.asarray(fields)


def minibatches(inputs, fields, batch_size, key):
    """Go once through the samples in random order, batch_size samples at a time."""
    order = jax.random.permutation(key, len(inputs))
    for start in range(0, len(inputs), batch_size):
        batch = order[start : start + batch_size]
        yield inputs[batch], fields[batch]
