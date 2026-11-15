"""Train the DRE on the training set and save it in the results folder.

The training samples are the pairs of consecutive states of each scenario. Three terms
are minimized together:
reconstruction: the autoencoder must reproduce the states it receives
dynamics:       the dynamics network must give the change of the latent variables
                between the two states, divided by the time step
regularization: sum of the absolute values of the weights
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import dre
from src.data.dataset import load_split

SEED = 0
N_EPOCHS = 501
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCH_OF_REDUCED_LEARNING_RATE = 100  # the learning rate is divided by 10 from here on
REGULARIZATION_WEIGHT = 1e-9


def load_pairs(split, scaling):
    """Scaled parameters, state and next state of every pair of consecutive states."""
    parameters, _, fields = load_split(split)
    states = dre.scale_state(scaling, fields)
    n_scenarios, n_times, _ = states.shape
    parameters_scaled = np.repeat(dre.scale_parameters(parameters), n_times - 1, axis=0)
    states_old = states[:, :-1].reshape(n_scenarios * (n_times - 1), -1)
    states_new = states[:, 1:].reshape(n_scenarios * (n_times - 1), -1)
    return parameters_scaled.astype(np.float32), states_old, states_new


def mse(a, b):
    return jnp.mean((a - b) ** 2)


def losses(model, parameters_scaled, states_old, states_new):
    latent_old = dre.encode(model, states_old)
    latent_new = dre.encode(model, states_new)
    reconstruction = mse(dre.decode(model, latent_old, training=True), states_old)
    reconstruction += mse(dre.decode(model, latent_new, training=True), states_new)

    # this term trains the dynamics network only: the latent variables are taken as data
    latent_old = jax.lax.stop_gradient(latent_old)
    latent_new = jax.lax.stop_gradient(latent_new)
    derivative = dre.latent_derivative(model, parameters_scaled, latent_old)
    dynamics = mse(latent_new - latent_old, dre.TIME_STEP * derivative)
    return reconstruction, dynamics


def regularization(model):
    absolute_weights = [jnp.abs(weights).sum() for weights in jax.tree_util.tree_leaves(model)]
    return REGULARIZATION_WEIGHT * sum(absolute_weights)


def total_loss(model, parameters_scaled, states_old, states_new):
    reconstruction, dynamics = losses(model, parameters_scaled, states_old, states_new)
    return reconstruction + dynamics + regularization(model), (reconstruction, dynamics)


@jax.jit
def training_step(model, optimizer_state, parameters_scaled, states_old, states_new):
    gradients, terms = jax.grad(total_loss, has_aux=True)(
        model, parameters_scaled, states_old, states_new
    )
    updates, optimizer_state = optimizer.update(gradients, optimizer_state)
    return optax.apply_updates(model, updates), optimizer_state, terms


_, _, training_fields = load_split("train")
scaling = dre.compute_scaling(training_fields)
training_pairs = load_pairs("train", scaling)
validation_pairs = load_pairs("validation", scaling)
n_pairs = len(training_pairs[0])
n_batches = int(np.ceil(n_pairs / BATCH_SIZE))

model = dre.new_model(jax.random.key(SEED))
learning_rate = optax.piecewise_constant_schedule(
    LEARNING_RATE, {EPOCH_OF_REDUCED_LEARNING_RATE * n_batches: 0.1}
)
optimizer = optax.adam(learning_rate)
optimizer_state = optimizer.init(model)
validation_losses = jax.jit(losses)
random = np.random.default_rng(SEED)

history = []
start = time.time()

for epoch in range(1, N_EPOCHS + 1):
    order = random.permutation(n_pairs)
    training_losses = []
    for first in range(0, n_pairs, BATCH_SIZE):
        batch = order[first : first + BATCH_SIZE]
        model, optimizer_state, terms = training_step(
            model, optimizer_state, *[pairs[batch] for pairs in training_pairs]
        )
        training_losses.append(terms)

    training_losses = np.mean(np.array(training_losses), axis=0).tolist()
    validation = np.array(validation_losses(model, *validation_pairs)).tolist()
    history.append([epoch, *training_losses, *validation])

    if epoch % 50 == 1:
        print(
            f"epoch {epoch}:"
            f" training loss {sum(training_losses):.2e},"
            f" validation loss {sum(validation):.2e}"
        )

print(f"training time: {time.time() - start:.0f} s")

os.makedirs("results", exist_ok=True)
dre.save_model(model, scaling, "results/dre.pkl")
np.savetxt(
    "results/dre_losses.txt",
    np.array(history),
    header="epoch training_reconstruction training_dynamics"
    " validation_reconstruction validation_dynamics",
)
