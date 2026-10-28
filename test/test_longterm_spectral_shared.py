"""Spectral shared-process regressions; no formal inputs or MICA sampling."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import yaml

HERE = Path(__file__).resolve().parent / "mica_recovery/longterm_test"
sys.path.insert(0, str(HERE))
import prepare


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load_module("spectral_fit_tests", "fit.py")
plots = load_module("spectral_plot_tests", "plot_inputs.py")


class SpectralSharedTests(unittest.TestCase):
    def setUp(self):
        self.config = prepare.load_config(HERE / "experiment_spectral.yaml")
        self.name, self.p = next(iter(prepare.spectral_shared_cases(self.config).items()))
        self.tw = np.array([0.0, 1.1, 9.0, 41.0, 100.0, 300.0])
        self.tu = np.array([0.5, 5.0, 30.0, 43.0, 150.0, 310.0, 420.0])

    def test_gaussian_ou_covariance_matches_direct_integration(self):
        for delta, sigma, tau in ((0., 80., 40.), (31., 80., 40.), (-3., .1, 40.)):
            expected = quad(lambda z: np.exp(-abs(delta - z) / tau - z*z/(2*sigma*sigma))
                            / (np.sqrt(2*np.pi)*sigma), -12*sigma, 12*sigma,
                            points=[delta] if abs(delta) < 12*sigma else None,
                            epsabs=1e-11)[0]
            self.assertAlmostEqual(float(prepare.gaussian_ou_covariance(delta, sigma, tau)), expected, places=10)

    def test_joint_covariance_matches_expected_likelihood_example(self):
        p = self.p
        nw, nu = len(self.tw), len(self.tu)
        c = prepare.spectral_component_covariance(self.tw, self.tu, p)
        np.testing.assert_allclose(c, c.T, atol=1e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(c).min(), -1e-12)
        np.testing.assert_array_equal(c[:nw, nw:2*nw], 0)
        # Map latent component samples to [W, U].
        a = np.zeros((nw + nu, 2*nw + 2*nu))
        a[:nw, :nw] = a[:nw, nw:2*nw] = np.eye(nw)
        a[nw:, 2*nw:2*nw+nu] = p['narrow_gain'] * np.eye(nu)
        a[nw:, 2*nw+nu:] = p['slow_gain'] * np.eye(nu)
        actual = a @ c @ a.T
        v = p['driver_stationary_rms']**2
        k = lambda d, s: v * prepare.gaussian_ou_covariance(d, s, p['driver_tau_days'])
        dx = self.tw[:, None] - self.tw
        dy = self.tu[:, None] - self.tu
        cross = self.tu[None, :] - self.tw[:, None]
        ad, ag = p['narrow_gain'], p['slow_gain']
        md, mg = p['narrow_mean_days'], p['lag_days']
        sd, sg = p['narrow_sigma_days'], p['slow_sigma_days']
        np.testing.assert_allclose(actual[:nw, :nw], k(dx, 0), atol=1e-14)
        np.testing.assert_allclose(actual[:nw, nw:], ad*k(cross-md, sd)+ag*k(cross-mg, sg), atol=1e-14)
        deterministic = (ad**2*k(dy, np.sqrt(2)*sd) + ag**2*k(dy, np.sqrt(2)*sg)
                         + ad*ag*(k(dy-(mg-md), np.hypot(sd, sg))
                                  + k(dy+(mg-md), np.hypot(sd, sg))))
        residual = ag**2*(k(dy, sg)-k(dy, np.sqrt(2)*sg))
        np.testing.assert_allclose(actual[nw:, nw:], deterministic+residual, atol=1e-14)
        self.assertGreater(np.diag(residual).min(), 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(residual).min(), -1e-12)

    def test_sampler_reconstructs_covariance_without_jitter(self):
        tw, tu = self.tw[:2], self.tu[:2]
        c = prepare.spectral_component_covariance(tw, tu, self.p)
        columns = []
        for basis in np.eye(len(c)):
            class BasisRng:
                def standard_normal(self, size):
                    return basis
            columns.append(np.concatenate(prepare.sample_spectral_components(tw, tu, self.p, BasisRng())))
        factor = np.column_stack(columns)
        np.testing.assert_allclose(factor @ factor.T, c, atol=1e-13)

    def test_generation_plotting_and_native_fit_plan(self):
        times = {'UVW2': self.tw, 'U': self.tu}
        errors = {band: np.full(len(t), 0.01) for band, t in times.items()}
        setup = ('UVW2', 'U', 59000., times, errors, self.p['driver_stationary_rms'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yaml'
            path.write_text(yaml.safe_dump(self.config))
            with patch.object(prepare, 'observation_setup', return_value=setup), redirect_stdout(io.StringIO()):
                prepare.prepare_spectral_shared(path)
                with self.assertRaises(FileExistsError):
                    prepare.prepare_spectral_shared(path)
            case = Path(directory) / 'cases' / self.name
            inputs = {}
            for band in times:
                inputs[band] = np.loadtxt(case / 'input' / f'LTSIM_{band}.txt')
                parts = plots.load_components(case, band, inputs[band], spectral_case=self.p)
                np.testing.assert_allclose(inputs[band][:, 0], times[band])
                np.testing.assert_allclose(inputs[band][:, 2], errors[band])
                np.testing.assert_allclose(parts[:, 1:3].mean(axis=0), 0, atol=1e-12)
                if band == 'UVW2':
                    self.assertGreater(abs(np.std(parts[:, 1:3].sum(axis=1), ddof=1)
                                           - self.p['driver_stationary_rms']), 1e-3)
            changed = {**self.p, 'lag_days': 31.}
            with self.assertRaisesRegex(ValueError, 'Configuration does not match'):
                plots.load_components(case, 'U', inputs['U'], spectral_case=changed)
            with patch.object(prepare, 'observation_setup', return_value=setup), redirect_stdout(io.StringIO()):
                prepare.prepare_spectral_shared(path, overwrite=True)
            for band in times:
                np.testing.assert_array_equal(np.loadtxt(case / 'input' / f'LTSIM_{band}.txt'), inputs[band])
            plan = fit.build_plan(path, suite='spectral_shared', model='gaussian')
            self.assertEqual(len(plan), 1)
            stage = plan[0]['fit']
            self.assertEqual(stage['component_order'], 'native')
            self.assertEqual(stage['type_lag_prior'], 0)
            self.assertEqual(stage['lag_prior'], [[0., 100.], [0., 100.]])
            np.testing.assert_allclose(stage['width_prior'], [self.config['mica_fit']['sigma_days']]*2)
            data = {'set1': [inputs['UVW2'], inputs['U']]}
            native = fit.run_mica.setup_kwargs(stage, data, 'U')
            self.assertNotIn('lag_prior', native)
            self.assertEqual(native['type_lag_prior'], 0)
            self.assertEqual(fit.run_mica.ordered_gaussian_prior_text(stage, data, 'U'),
                             '5 0-th_component_centroid UNI 0 100 0 0\n8 1-th_component_centroid UNI 0 100 0 0\n')
            self.assertFalse(stage['fit_only'])
            self.assertFalse(stage['flag_negative_resp'])
            self.assertTrue(stage['plot_decomp'])
            with patch.object(plots.plt.Figure, 'savefig') as save, redirect_stdout(io.StringIO()):
                plots.plot_case(case, 'LTSIM', ('UVW2', 'U'), spectral_case=self.p)
                lag, kernels = plots.build_transfer_functions(self.config)
                plots.plot_transfer_function(case, self.name, lag, kernels[self.name])
            self.assertEqual(save.call_count, 4)

    def test_effective_kernel_has_correct_areas_and_negative_tail(self):
        lag, curves = plots.build_transfer_functions(self.config)
        curve = curves[self.name]
        dt = lag[1] - lag[0]
        self.assertIn('not a deterministic', curve['title'])
        self.assertNotIn('disk', curve)
        self.assertAlmostEqual(curve['narrow'].sum()*dt, .7, places=9)
        self.assertAlmostEqual(curve['gaussian'].sum()*dt, .3, places=9)
        self.assertAlmostEqual(curve['gaussian'][lag < 0].sum()*dt/.3, ndtr(-30/80), places=9)

    def test_legacy_fit_order_and_configuration_are_unchanged(self):
        for filename in ('experiment.yaml', 'experiment_broad.yaml'):
            cfg = fit.source_config(prepare.load_config(HERE / filename))
            self.assertEqual(cfg['mica_gaussian2']['type_lag_prior'], 4)
            self.assertNotIn('component_order', cfg['mica_gaussian2'])
        bad = deepcopy(self.config)
        bad['spectral_shared']['driver_stationary_rms'] = -1
        with self.assertRaises(ValueError):
            prepare.spectral_shared_cases(bad)


if __name__ == '__main__':
    unittest.main()
