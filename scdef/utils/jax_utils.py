import numpy as np
import jax.numpy as jnp
from jax import vmap, random
from jax.scipy.stats import norm, gamma, poisson


def gaussian_sample(rng, mean, log_scale):
    scale = jnp.exp(log_scale)
    return mean + scale * random.normal(rng, mean.shape)


def gaussian_logpdf(x, mean, log_scale):
    scale = jnp.exp(log_scale)
    return jnp.sum(
        vmap(norm.logpdf)(x, mean * jnp.ones(x.shape), scale * jnp.ones(x.shape))
    )


def gamma_sample(rng, shape, rate):
    scale = 1.0 / rate
    return jnp.clip(scale * random.gamma(rng, shape), a_min=1e-15, a_max=1e15)


def gamma_logpdf(x, shape, rate):
    scale = 1.0 / rate
    return jnp.sum(
        vmap(gamma.logpdf)(
            x, shape * jnp.ones(x.shape), scale=scale * jnp.ones(x.shape)
        )
    )
