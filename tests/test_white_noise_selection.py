"""Numerical equivalence and identifiability checks for backend screening."""

import unittest
import numpy as np
import whitenoise_estimator as wn
from pleb.white_noise_selection import (
    design_basis,
    restricted_nll_gradient,
    screen_backend,
)


class BackendNoiseTests(unittest.TestCase):
    def test_matches_whitenoise_dense_restricted_likelihood(self):
        rng = np.random.default_rng(42)
        n = 32
        r = rng.normal(size=n) * 2e-6
        sigma = rng.uniform(0.3, 2, n) * 1e-6
        m = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
        basis = design_basis(m)
        epochs = np.repeat(np.arange(16), 2)
        x = np.log([1.3, 0.6, 0.9])
        value, _ = restricted_nll_gradient(x, ("F", "Q", "J"), r, sigma, basis, epochs)
        g = wn._null_space_basis(m)

        def builder(parameters):
            return wn._build_covariance_multi(
                parameters, sigma, wn._epoch_membership_matrix(epochs)
            )

        dense = wn._restricted_negative_log_likelihood(
            np.log([1.3, 0.6e-6, 0.9e-6]), r, g, builder, 1e-30
        )
        self.assertAlmostEqual(value, dense, places=8)

    def test_analytic_gradient(self):
        rng = np.random.default_rng(43)
        n = 30
        r = rng.normal(size=n) * 1e-6
        sigma = rng.uniform(0.5, 1.5, n) * 1e-6
        basis = design_basis(np.column_stack([np.ones(n), np.linspace(-1, 1, n)]))
        epochs = np.repeat(np.arange(10), 3)
        x = np.log([1.1, 0.3, 0.4])
        _, gradient = restricted_nll_gradient(
            x, ("F", "Q", "J"), r, sigma, basis, epochs
        )
        numeric = []
        for i in range(3):
            delta = np.eye(3)[i] * 1e-5
            a = restricted_nll_gradient(
                x + delta, ("F", "Q", "J"), r, sigma, basis, epochs
            )[0]
            b = restricted_nll_gradient(
                x - delta, ("F", "Q", "J"), r, sigma, basis, epochs
            )[0]
            numeric.append((a - b) / 2e-5)
        np.testing.assert_allclose(gradient, numeric, rtol=1e-5, atol=1e-6)

    def test_single_epochs_and_constant_errors_are_not_separately_identifiable(self):
        rng = np.random.default_rng(44)
        n = 35
        context, models, summary = screen_backend(
            rng.normal(size=n) * 1e-6,
            np.ones(n) * 1e-6,
            np.arange(n) * 100.0,
            np.ones((n, 1)),
        )
        self.assertFalse(context["ecorr_structurally_identifiable"])
        self.assertFalse(context["efac_equad_structurally_identifiable"])
        self.assertTrue(all("J" not in x["model"] for x in models))
        self.assertTrue(all(x["model"] != "F+Q" for x in models))

    def test_recovers_injected_epoch_correlations(self):
        rng = np.random.default_rng(45)
        n = 240
        epochs = np.repeat(np.arange(80), 3)
        sigma = rng.uniform(0.4, 1.8, n) * 1e-6
        r = rng.normal(size=n) * sigma + rng.normal(size=80)[epochs] * 4e-6
        context, models, summary = screen_backend(
            r, sigma, epochs * 100.0 + np.tile([0.0, 0.1, 0.2], 80), np.ones((n, 1))
        )
        self.assertTrue(context["ecorr_structurally_identifiable"])
        self.assertTrue(summary["ecorr_supported_BIC6"])
        self.assertGreater(summary["best_ecorr_us"], 2.0)
        self.assertLess(summary["best_ecorr_us"], 6.0)


if __name__ == "__main__":
    unittest.main()
