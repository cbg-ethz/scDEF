import numpy as np
import jax.numpy as jnp
from jax import vmap
import tensorflow_probability.substrates.jax.distributions as tfd


def gaussian_sample(rng, mean, log_scale):
    scale = jnp.exp(log_scale)
    return mean + scale * random.normal(rng, mean.shape)


def gaussian_logpdf(x, mean, log_scale):
    scale = jnp.exp(log_scale)
    return jnp.sum(
        vmap(norm.logpdf)(x, mean * jnp.ones(x.shape), scale * jnp.ones(x.shape))
    )


def gamma_sample(rng, shape, rate):
    return tfd.Gamma(shape, rate).sample(seed=rng)


def _gamma_logpdf(x, shape, rate):
    return tfd.Gamma(shape, rate).log_prob(x)


def gamma_logpdf(x, shape, rate):
    shape = shape * jnp.ones(x.shape)
    rate = rate * jnp.ones(x.shape)
    return jnp.sum(
        vmap(vmap(_gamma_logpdf, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(x, shape, rate)
    )


def _gamma_entropy(shape, rate):
    return tfd.Gamma(shape, rate).entropy()


def gamma_entropy(shape, rate):
    return jnp.sum(
        vmap(vmap(_gamma_entropy, in_axes=(0, 0)), in_axes=(0, 0))(shape, rate)
    )
