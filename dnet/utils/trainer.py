from typing import List

from jax import jit, grad
from jax.experimental.stax import serial
from tqdm import tqdm

from ..optimizers import OptimizerState


def train(config: dict, inputs, targets):
    seed = config.get("_seed")
    epochs = config.get("_epochs")
    loss_fn = config.get("_loss_fn")
    trained_params = config.get("_trained_params")
    opt_init, opt_update, fetch_params = config.get("_optimizer")
    setup_params, forward_pass = serial(*config.get("_layers"))

    def initialize_params():
        from jax.random import PRNGKey
        rng = PRNGKey(seed)
        input_shape = list(inputs.shape)
        input_shape[0] = -1
        input_shape = tuple(input_shape)
        _, params = setup_params(rng=rng, input_shape=input_shape)
        if trained_params:
            params = trained_params
        return params

    network_params = initialize_params()

    def begin_training():
        losses: List[float] = []
        opt_state: OptimizerState = opt_init(network_params)
        progress_bar = tqdm(iterable=range(epochs), desc="Training model", leave=True)
        for i in progress_bar:
            opt_state = step(i, opt_state)
            params = fetch_params(opt_state)
            losses.append(compute_loss(params).item())
            progress_bar.set_postfix_str(f"Loss : {losses[-1]}")
            progress_bar.refresh()
        config["_metrics"] = {"losses": losses}
        config["_trained_params"] = fetch_params(opt_state)

    @jit
    def step(i, opt_state):
        params = fetch_params(opt_state)
        grads = grad(compute_loss)(params)
        return opt_update(i, grads, opt_state)

    @jit
    def compute_loss(params):
        predictions = forward_pass(params, inputs)
        return jit(loss_fn)(predictions, targets)

    begin_training()
    return config
