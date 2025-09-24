import jax.numpy as jnp
from jax import vmap
import tensorflow_probability.substrates.jax.distributions as tfd


def lognormal_sample(rng, m, s):
    """Sample from a log-normal distribution with clipping."""
    return jnp.clip(tfd.LogNormal(m, s).sample(seed=rng), 1e-15, 1e15)


def _lognormal_entropy(m, s):
    """Compute entropy of log-normal distribution."""
    return tfd.LogNormal(m, s).entropy()


def lognormal_entropy(m, s):
    """Compute entropy of log-normal distribution with vectorization."""
    return jnp.sum(vmap(vmap(_lognormal_entropy, in_axes=(0, 0)), in_axes=(0, 0))(m, s))


def _lognormal_logpdf(x, m, s):
    """Compute log-probability density of log-normal distribution."""
    return tfd.LogNormal(m, s).log_prob(x)


def lognormal_logpdf(x, m, s):
    """Compute log-probability density of log-normal distribution with vectorization."""
    m = m * jnp.ones(x.shape)
    s = s * jnp.ones(x.shape)
    return jnp.sum(
        vmap(vmap(_lognormal_logpdf, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(x, m, s)
    )


def _gamma_logpdf(x, shape, rate):
    """Compute log-probability density of gamma distribution."""
    return tfd.Gamma(shape, rate).log_prob(x)


def gamma_logpdf(x, shape, rate):
    """Compute log-probability density of gamma distribution with vectorization."""
    shape = shape * jnp.ones(x.shape)
    rate = rate * jnp.ones(x.shape)
    return jnp.sum(
        vmap(vmap(_gamma_logpdf, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(x, shape, rate)
    )
