"""Dynamical Reduced Embedding (DRE) of the pressure and saturation fields.

The state is the vector of the fields at one time: the pressure on the cells followed by
the saturation on the cells. An autoencoder compresses the state into a few latent
variables, and the dynamics network gives the time derivative of the latent variables
from the parameters and the latent variables themselves. A prediction encodes the
initial state, advances the latent variables in time with the explicit Euler method and
decodes them.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np

from src.data.dataset import N_CELLS

N_PARAMETERS = 2
LATENT_SIZE = 3
TIME_STEP = 0.02  # time between two consecutive states of the data

# ranges from which log(K45) and h are sampled
PARAMETERS_MIN = np.array([np.log(5e-3), -0.05])
PARAMETERS_MAX = np.array([np.log(2e3), 0.05])


def new_layers(sizes, key):
    """Random weights of fully connected layers of the given sizes."""
    layers = []
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        key, layer_key = jax.random.split(key)
        weights = jax.random.normal(layer_key, (size_in, size_out)) * np.sqrt(2 / size_in)
        layers.append({"weights": weights, "bias": jnp.zeros(size_out)})
    return layers


def new_model(key):
    """Encoder, decoder and dynamics network with random weights.

    The slopes are those of the PReLU activations, which are trained with the weights.
    """
    key_encoder, key_decoder, key_dynamics = jax.random.split(key, 3)
    dynamics_sizes = [N_PARAMETERS + LATENT_SIZE, 20, 20, 20, 20, 20, LATENT_SIZE]
    return {
        "encoder": new_layers([2 * N_CELLS, LATENT_SIZE], key_encoder),
        "decoder": new_layers([LATENT_SIZE, 100, 100, 2 * N_CELLS], key_decoder),
        "decoder_slopes": jnp.full(3, 0.25),
        "dynamics": new_layers(dynamics_sizes, key_dynamics),
        "dynamics_slopes": jnp.full(6, 0.25),
    }


def linear(layer, x):
    return x @ layer["weights"] + layer["bias"]


def prelu(x, slope):
    return jnp.where(x >= 0, x, slope * x)


def soft_clip(x):
    """Like a clip to [0, 1], but with a slope of 0.1 for ten units outside the interval.

    It is used in the training, where the gradient of a clip would be zero outside.
    """
    return jnp.clip(x, 0, 1) + 0.1 * (jnp.clip(x, -10, 0) + jnp.clip(x, 1, 11) - 1)


def encode(model, state_scaled):
    """Latent variables of a scaled state."""
    return linear(model["encoder"][0], state_scaled)


def decode(model, latent, training=False):
    """Scaled state of some latent variables. The scaled saturation is kept in [0, 1]."""
    slopes = model["decoder_slopes"]
    hidden = prelu(linear(model["decoder"][0], latent), slopes[0])
    hidden = prelu(linear(model["decoder"][1], hidden), slopes[1])
    output = linear(model["decoder"][2], hidden)

    pressure = prelu(output[..., :N_CELLS], slopes[2])
    saturation = output[..., N_CELLS:]
    if training:
        saturation = soft_clip(saturation)
    else:
        saturation = jnp.clip(saturation, 0, 1)
    return jnp.concatenate([pressure, saturation], axis=-1)


def latent_derivative(model, parameters_scaled, latent):
    """Time derivative of the latent variables."""
    x = jnp.concatenate([parameters_scaled, latent], axis=-1)
    for layer, slope in zip(model["dynamics"], model["dynamics_slopes"]):
        x = prelu(linear(layer, x), slope)
    return x


def compute_scaling(training_fields):
    """Minimum and range of each value of the state over the training set."""
    minimum = training_fields.min(axis=(0, 1))
    value_range = training_fields.max(axis=(0, 1)) - minimum
    # a value that never changes is not scaled
    constant = value_range < 1e-7
    return {
        "state_min": np.where(constant, 0, minimum),
        "state_range": np.where(constant, 1, value_range),
    }


def scale_state(scaling, state):
    """Each value of the state to [0, 1], with its range over the training set."""
    return (state - scaling["state_min"]) / scaling["state_range"]


def unscale_state(scaling, state_scaled):
    return state_scaled * scaling["state_range"] + scaling["state_min"]


def scale_parameters(parameters):
    """log(K45) and h to [0, 1], with the ranges from which they are sampled."""
    parameters = np.asarray(parameters)
    log_parameters = np.stack([np.log(parameters[..., 0]), parameters[..., 1]], axis=-1)
    return (log_parameters - PARAMETERS_MIN) / (PARAMETERS_MAX - PARAMETERS_MIN)


@jax.jit
def euler_step(model, parameters_scaled, latent):
    return latent + TIME_STEP * latent_derivative(model, parameters_scaled, latent)


def predict(model, scaling, parameters, initial_state, n_times):
    """States of one scenario at n_times consecutive times, the first is the initial one."""
    parameters_scaled = jnp.asarray(scale_parameters(parameters))
    latent = encode(model, scale_state(scaling, initial_state))
    latents = [latent]
    for _ in range(n_times - 1):
        latent = euler_step(model, parameters_scaled, latent)
        latents.append(latent)
    states_scaled = decode(model, jnp.stack(latents))
    return np.asarray(unscale_state(scaling, states_scaled))


def n_trainable_parameters(model):
    return sum(weights.size for weights in jax.tree_util.tree_leaves(model))


def save_model(model, scaling, file_name):
    with open(file_name, "wb") as file:
        pickle.dump({"model": jax.device_get(model), "scaling": scaling}, file)


def load_model(file_name):
    with open(file_name, "rb") as file:
        saved = pickle.load(file)
    return saved["model"], saved["scaling"]
