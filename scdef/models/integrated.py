class Integrated(scDEF):
    """
    Different cells, overlapping genes between two modalities.
    """

    def elbo(self, rng, indices, var_params):
        # Single-sample Monte Carlo estimate of the variational lower bound.
        batch_indices_onehot = self.batch_indices_onehot[indices]

        min_loc = jnp.log(1e-10)
        cell_scale_params = jnp.clip(var_params[0], a_min=min_loc)
        gene_scale_params = jnp.clip(var_params[1], a_min=min_loc)
        #         unconst_gene_scales = jnp.clip(var_params[1], a_min=min_loc)
        hfactor_scale_params = jnp.clip(var_params[2], a_min=min_loc)
        factor_scale_params = jnp.clip(var_params[3], a_min=min_loc)
        hz_params = jnp.clip(var_params[4], a_min=min_loc)
        hW_params = jnp.clip(var_params[5], a_min=min_loc)
        #         unconst_hW = jnp.clip(var_params[5], a_min=min_loc)
        z_params = jnp.clip(var_params[6], a_min=min_loc)
        W_params = jnp.clip(var_params[7], a_min=min_loc)
        #         unconst_W = jnp.clip(var_params[7], a_min=min_loc)
        W_noise_params = jnp.clip(var_params[8], a_min=min_loc)
        z_noise_params = jnp.clip(var_params[9], a_min=min_loc)

        min_concentration = 1e-10
        min_scale = 1e-10
        cell_scale_concentration = jnp.maximum(
            jnn.softplus(cell_scale_params[0][indices]), min_concentration
        )
        cell_scale_rate = 1.0 / jnp.maximum(
            jnn.softplus(cell_scale_params[1][indices]), min_scale
        )

        z_concentration = jnp.maximum(
            jnn.softplus(z_params[0][indices]), min_concentration
        )
        z_rate = 1.0 / jnp.maximum(jnn.softplus(z_params[1][indices]), min_scale)

        hz_concentration = jnp.maximum(
            jnn.softplus(hz_params[0][indices]), min_concentration
        )
        hz_rate = 1.0 / jnp.maximum(jnn.softplus(hz_params[1][indices]), min_scale)

        gene_scale_concentration = jnp.maximum(
            jnn.softplus(gene_scale_params[0]), min_concentration
        )
        gene_scale_rate = 1.0 / jnp.maximum(
            jnn.softplus(gene_scale_params[1]), min_scale
        )

        hfactor_scale_concentration = jnp.maximum(
            jnn.softplus(hfactor_scale_params[0]), min_concentration
        )
        hfactor_scale_rate = 1.0 / jnp.maximum(
            jnn.softplus(hfactor_scale_params[1]), min_scale
        )

        factor_scale_concentration = jnp.maximum(
            jnn.softplus(factor_scale_params[0]), min_concentration
        )
        factor_scale_rate = 1.0 / jnp.maximum(
            jnn.softplus(factor_scale_params[1]), min_scale
        )

        W_concentration = jnp.maximum(jnn.softplus(W_params[0]), min_concentration)
        W_rate = 1.0 / jnp.maximum(jnn.softplus(W_params[1]), min_scale)

        hW_concentration = jnp.maximum(jnn.softplus(hW_params[0]), min_concentration)
        hW_rate = 1.0 / jnp.maximum(jnn.softplus(hW_params[1]), min_scale)

        z_noise_concentration = jnp.maximum(
            jnn.softplus(z_noise_params[0][indices]), min_concentration
        )
        z_noise_rate = 1.0 / jnp.maximum(
            jnn.softplus(z_noise_params[1][indices]), min_scale
        )

        W_noise_concentration = jnp.maximum(
            jnn.softplus(W_noise_params[0]), min_concentration
        )
        W_noise_rate = 1.0 / jnp.maximum(jnn.softplus(W_noise_params[1]), min_scale)

        # Sample from variational distribution
        cell_scales = gamma_sample(rng, cell_scale_concentration, cell_scale_rate)
        #         cell_scales = jnp.exp(log_cell_scales)
        gene_scales = gamma_sample(rng, gene_scale_concentration, gene_scale_rate)
        #         gene_scales = jnp.exp(log_gene_scales)
        #         gene_scales = jnn.softplus(unconst_gene_scales)

        #         log_hfactor_scales = gaussian_sample(rng, hfactor_scale_params[0], hfactor_scale_params[1])
        #         hfactor_scales = jnp.exp(log_hfactor_scales)
        hfactor_scales = gamma_sample(
            rng, hfactor_scale_concentration, hfactor_scale_rate
        )
        factor_scales = gamma_sample(rng, factor_scale_concentration, factor_scale_rate)
        #         factor_scales = jnp.exp(log_factor_scales)

        hz = gamma_sample(rng, hz_concentration, hz_rate)
        #         hz = jnp.exp(log_hz)
        hW = gamma_sample(rng, hW_concentration, hW_rate)
        #         hW = jnp.exp(log_hW)
        #         hW = jnn.softplus(unconst_hW)
        mean_top = jnp.matmul(hz, hW)

        z = gamma_sample(rng, z_concentration, z_rate)
        #         z = jnp.exp(log_z)
        #         log_z_noise = gaussian_sample(rng, z_noise_params[0][indices], z_noise_params[1][indices])
        #         z_noise = jnp.exp(log_z_noise)
        #         z_noise = gamma_sample(rng, z_noise_concentration, z_noise_rate)
        W = gamma_sample(rng, W_concentration, W_rate)
        #         W = jnp.exp(log_W)
        #         W = jnn.softplus(unconst_W)
        #         W_noise = jnn.softplus(unconst_W_noise)
        #         W_noise = gamma_sample(rng, W_noise_concentration, W_noise_rate)
        mean_bottom_bio = jnp.matmul(z, W)
        #         mean_bottom_batch = jnp.matmul(batch_indices_onehot * z_noise, W_noise) # jnn.softplus(jnp.matmul(batch_indices_onehot, unconst_W_noise))
        mean_bottom = mean_bottom_bio  # + mean_bottom_batch

        # Compute log likelihood
        ll = jnp.sum(vmap(poisson.logpmf)(self.X[indices], mean_bottom))

        # Compute KL divergence
        kl = 0.0
        gene_size = jnp.sum(self.X, axis=0)
        kl += gamma_logpdf(
            gene_scales, 1.0, jnp.mean(gene_size) / jnp.var(gene_size)
        ) - gamma_logpdf(gene_scales, gene_scale_concentration, gene_scale_rate)

        kl += gamma_logpdf(hfactor_scales, 1e-3, 1e-3) - gamma_logpdf(
            hfactor_scales, hfactor_scale_concentration, hfactor_scale_rate
        )

        kl += gamma_logpdf(factor_scales, 1e-3, 1e-3) - gamma_logpdf(
            factor_scales, factor_scale_concentration, factor_scale_rate
        )
        #         normalized_factor_scales = factor_scales / jnp.sum(factor_scales)

        kl += gamma_logpdf(hW, 0.3, 1.0 * hfactor_scales.reshape(-1, 1)) - gamma_logpdf(
            hW, hW_concentration, hW_rate
        )

        kl += gamma_logpdf(
            W, 0.3, 1 * gene_scales.reshape(1, -1) * factor_scales.reshape(-1, 1)
        ) - gamma_logpdf(W, W_concentration, W_rate)

        #         kl += gamma_logpdf(W_noise, 10, 10. * gene_scales.reshape(1,-1)) -\
        #                 gamma_logpdf(W_noise, W_noise_concentration, W_noise_rate)

        kl *= indices.shape[0] / self.X.shape[0]  # scale by minibatch size

        kl += gamma_logpdf(
            cell_scales, 1.0, 1.0 * self.batch_lib_ratio[indices]
        ) - gamma_logpdf(cell_scales, cell_scale_concentration, cell_scale_rate)

        kl += gamma_logpdf(hz, 1.0, 1.0 / hfactor_scales.reshape(1, -1)) - gamma_logpdf(
            hz, hz_concentration, hz_rate
        )

        # Tieing the scales avoids the factors with low scales in W learn z's that correlate with cell scales
        kl += gamma_logpdf(
            z,
            self.shape,
            cell_scales.reshape(-1, 1)
            * self.shape
            / (factor_scales.reshape(1, -1) * mean_top),
        ) - gamma_logpdf(z, z_concentration, z_rate)

        #         kl += gamma_logpdf(z_noise, 10., 10. * cell_scales.reshape(-1,1)) -\
        #                 gamma_logpdf(z_noise, z_noise_concentration, z_noise_rate)

        return ll + kl
