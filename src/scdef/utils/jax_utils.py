import numpy as np
import jax.numpy as jnp
from jax import vmap
import tensorflow_probability.substrates.jax.distributions as tfd



def gamma_sample(rng, shape, rate):
    scale = 1.0 / rate
    return jnp.clip(
        scale * jnp.exp(jax.random.loggamma(rng, shape)), a_min=1e-15, a_max=1e15
    )

def _gamma_logpdf(x, shape, rate):
    return tfd.Gamma(shape, rate).log_prob(x)
def gamma_logpdf(x, shape, rate):
    shape = shape * jnp.ones(x.shape)
    rate = rate * jnp.ones(x.shape)
    return jnp.sum(vmap(vmap(_gamma_logpdf, in_axes=(0,0,0)), in_axes=(0,0,0))(x, shape, rate))

def _gamma_entropy(shape, rate):
    return tfd.Gamma(shape, rate).entropy()
def gamma_entropy(shape,rate):
    return jnp.sum(vmap(vmap(_gamma_entropy, in_axes=(0,0)), in_axes=(0,0))(shape,rate))
