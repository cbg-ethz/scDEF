import jax
import jax.numpy as jnp
from jax import vmap
import tensorflow_probability.substrates.jax.distributions as tfd


def gamma_sample(rng, shape, rate):
    scale = 1.0 / rate
    return jnp.clip(tfd.Gamma(shape, rate).sample(seed=rng), a_min=1e-15, a_max=1e15)


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


def lognormal_sample(rng, m, s):
    return jnp.clip(tfd.LogNormal(m, s).sample(seed=rng), 1e-15, 1e15)


def _lognormal_entropy(m, s):
    return tfd.LogNormal(m, s).entropy()


def lognormal_entropy(m, s):
    return jnp.sum(vmap(vmap(_lognormal_entropy, in_axes=(0, 0)), in_axes=(0, 0))(m, s))


def _lognormal_logpdf(x, m, s):
    return tfd.LogNormal(m, s).log_prob(x)


def lognormal_logpdf(x, m, s):
    m = m * jnp.ones(x.shape)
    s = s * jnp.ones(x.shape)
    return jnp.sum(
        vmap(vmap(_lognormal_logpdf, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(x, m, s)
    )
