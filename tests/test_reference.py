import unittest

import numpy as np

from sord import SORDConfig, sord, sord_batch
from sord import steps


class GaussianPrior:
    """Toy prior x0 ~ N(0.5, s0^2) per pixel, with the paper's 120-level VE ladder.
    Its Tweedie denoiser is exact: E[x0 | x] = 0.5 + s0^2 / (s0^2 + sigma^2) * (x - 0.5)."""

    def __init__(self, s0=0.2):
        t = np.linspace(1, 1e-3, 121)[:-1]
        self.sigmas = np.sqrt((25.0 ** (2 * t) - 1) / (2 * np.log(25.0)))
        self.s0 = s0

    def denoise(self, x, i):
        g = self.s0 ** 2 / (self.s0 ** 2 + self.sigmas[i] ** 2)
        return 0.5 + g * (np.asarray(x) - 0.5)


class GaussianAudioPrior:
    """Toy continuous-sigma prior x0 ~ N(0, s0^2) per sample: E[x0 | x] = s0^2 / (s0^2 + sigma^2) * x."""

    def __init__(self, s0=0.2):
        self.s0 = s0

    def x0(self, x, sigma):
        return self.s0 ** 2 / (self.s0 ** 2 + sigma ** 2) * np.asarray(x)


def bench(seed=0, size=16):
    rng = np.random.default_rng(seed)
    x = np.clip(0.5 + 0.2 * rng.standard_normal((size, size, 3)), 0, 1)
    omega = np.zeros((size, size), bool)
    omega[4:10, 5:12] = True
    y = x + 0.3 * omega[..., None]
    return x, y, omega


class SORDTests(unittest.TestCase):
    def setUp(self):
        self.prior = GaussianPrior()
        self.cfg = SORDConfig(K=4, M=3, corruption="pos", tau_stop=0.01)

    def test_outside_support_is_exact_and_seeded(self):
        x, y, omega = bench()
        a = sord(y, omega, self.prior, self.cfg, seed=7)
        b = sord(y, omega, self.prior, self.cfg, seed=7)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_allclose(a.x[~omega], y[~omega])
        for it in a.iterates:
            np.testing.assert_allclose(it[~omega], y[~omega])

    def test_batch_does_not_change_an_images_result(self):
        imgs = [bench(seed=s) for s in range(3)]
        Y = np.stack([y for _, y, _ in imgs])
        OM = np.stack([o for _, _, o in imgs])
        OM[1, :, :] = False
        OM[1, 2:12, 3:7] = True                                    # a different support per image
        batched = sord_batch(Y, OM, self.prior, self.cfg, seeds=[11, 12, 13])
        for b in range(3):
            alone = sord(Y[b], OM[b], self.prior, self.cfg, seed=11 + b)
            np.testing.assert_allclose(batched[b].x, alone.x)
            self.assertEqual(batched[b].returned, alone.returned)
            np.testing.assert_allclose(batched[b].sigma_star, alone.sigma_star)

    def test_x_step_noise_matches_first_denoising_level(self):
        """The x-step must re-noise to the ladder level it starts denoising from (not to sigma_m itself)."""
        x, y, omega = bench()
        seen = []
        prior = self.prior

        class Spy:
            sigmas = prior.sigmas

            def denoise(self, z, i):
                seen.append(i)
                return prior.denoise(z, i)
        steps.x_step(Spy(), y[None], np.zeros_like(y)[None], omega[None], 0.16, 2, [np.random.default_rng(0)])
        self.assertEqual(seen[0], steps.levels_at_or_below(prior, 0.16)[0])

    def test_sigma_star_is_per_image_rms_spread_on_support(self):
        rng = np.random.default_rng(1)
        samples = rng.standard_normal((5, 8, 8, 3))
        omega = np.zeros((8, 8), bool)
        omega[2:5, 3:6] = True
        expect = np.sqrt(samples.var(0, ddof=1).mean(-1)[omega].mean())
        self.assertAlmostEqual(steps.sigma_star(samples[None], omega[None])[0], expect)

    def test_large_tau_returns_inpainting(self):
        x, y, omega = bench()
        r = sord(y, omega, self.prior, SORDConfig(K=4, M=3, tau_stop=1.0), seed=3)
        self.assertTrue(r.stopped_early)
        self.assertEqual(r.returned, 0)
        np.testing.assert_array_equal(r.x, r.iterates[0])

    def test_line6_discards_unproductive_refinement(self):
        x, y, omega = bench()
        r = sord(y, omega, self.prior, SORDConfig(K=4, M=4, tau_stop=0.0), seed=3)
        s = r.sigma_star                                           # spread of x_0, x_1, ...
        for m in range(1, r.returned + 1):
            self.assertGreater(s[m - 1] - s[m], 0.0)              # every accepted refinement reduced sigma*
        if r.stopped_early:                                        # the discarded one did not
            self.assertLessEqual(s[r.returned] - s[r.returned + 1], 0.0)
            self.assertEqual(len(r.iterates), r.returned + 2)

    def test_corruption_class_box(self):
        x, y, omega = bench()
        rng = np.random.default_rng(0)
        n = steps.n_step(self.prior, y[None], np.zeros_like(y)[None], omega[None], [0.05], [rng], kind="pos")[0]
        self.assertGreaterEqual(n.min(), 0.0)                      # n >= 0 for the "pos" class
        self.assertTrue(np.all(n <= y + 1e-12))
        self.assertEqual(np.abs(n[~omega]).max(), 0.0)             # only on Omega

    def test_shadow_step_lives_on_support_and_brightens(self):
        x, _, omega = bench()
        y = x * np.where(omega[..., None], 0.4, 1.0)               # shadow: multiply by 0.4 on Omega
        rng = np.random.default_rng(0)
        n = steps.h_step(self.prior, y[None], x[None], omega[None], [0.05], [rng])[0]
        self.assertEqual(np.abs(n[~omega]).max(), 0.0)
        self.assertTrue(np.all(n[omega] <= 1e-12))                 # y - n = y / h >= y

    def test_audio_outside_window_is_exact_and_batch_independent(self):
        rng = np.random.default_rng(0)
        L = 300
        Y = np.clip(0.2 * rng.standard_normal((2, L, 1)), -1, 1)
        OM = np.zeros((2, L), bool)
        OM[0, 100:200] = True
        OM[1, 40:260] = True
        Y[OM] += 0.3 * rng.standard_normal(OM.sum())[:, None]
        cfg = SORDConfig(modality="audio", K=4, M=3, sigma_start=0.32, sigma_end=0.08, tau_stop=0.01)
        prior = GaussianAudioPrior()
        batched = sord_batch(Y, OM, prior, cfg, seeds=[5, 6])
        for b in range(2):
            alone = sord(Y[b], OM[b], prior, cfg, seed=5 + b)
            np.testing.assert_allclose(batched[b].x, alone.x)
            np.testing.assert_allclose(batched[b].x[~OM[b]], Y[b][~OM[b]])

    def test_audio_large_tau_returns_inpainting(self):
        rng = np.random.default_rng(1)
        y = np.clip(0.2 * rng.standard_normal((200, 1)), -1, 1)
        om = np.zeros(200, bool)
        om[50:150] = True
        r = sord(y, om, GaussianAudioPrior(), SORDConfig(modality="audio", K=4, M=3, tau_stop=1.0), seed=2)
        self.assertEqual(r.returned, 0)
        np.testing.assert_array_equal(r.x, r.iterates[0])


if __name__ == "__main__":
    unittest.main()
