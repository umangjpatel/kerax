def BCELoss(inputs, outputs):
    import jax.numpy as jnp
    return - jnp.mean(a=outputs * jnp.log(inputs) + (1 - outputs) * jnp.log(1 - inputs))
