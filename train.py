import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data.datasets import create_uea_dataset
from models.RNN import create_rnn_model


@eqx.filter_jit
def calc_output(model, X):
    return jax.vmap(model)(X)


@eqx.filter_jit
@eqx.filter_value_and_grad
def classification_loss(model, X, y):
    pred_y = calc_output(model, X)
    return jnp.mean(-jnp.sum(y * jnp.log(pred_y + 1e-8), axis=1))


@eqx.filter_jit
def make_step(model, X, y, opt, opt_state):
    value, grads = classification_loss(model, X, y)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, value


key = jr.PRNGKey(1234)
datasetkey, batchkey, key = jr.split(key, 3)
num_steps = 1000
hidden_dim = 100
batch_size = 32

dataset = create_uea_dataset("LSST", use_idxs=False, key=datasetkey)
model = create_rnn_model(
    "linear", dataset.data_dim, hidden_dim, dataset.label_dim, key=key
)

opt = optax.adam(learning_rate=3e-4)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

running_loss = 0.0

for step, data in zip(
    range(num_steps), dataset.raw_dataloaders["train"].loop(batch_size, key=batchkey)
):
    X, y = data
    model, value = make_step(model, X, y, opt, opt_state)
    running_loss += value
    if (step + 1) % 100 == 0:
        print(f" Step: {step+1}, Loss: {running_loss/100}")
        running_loss = 0.0
