#!/usr/bin/env python3
"""Prepare five matched inputs for ten two-component MICA recovery fits.

Uses the saved tau_I=0.5 d realization. Only U is detrended; W2 is unchanged.
This script creates inputs/configuration and does not launch a sampler.
"""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import shutil
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'cases_fast_driver'
BASE = OUT / 'disk_blr_uonly_gp_tauI0p5'
CONFIG = HERE / 'experiment_uonly_gp_fast_driver.yaml'
SIGMAS = (5, 10, 20, 40)


def main():
    names = [BASE.name] + [f'{BASE.name}_u_s{s}' for s in SIGMAS]
    if CONFIG.exists() or any((OUT / name).exists() for name in names[1:]):
        raise FileExistsError('Recovery configuration or filtered inputs already exist; preserving them.')
    original_cfg = yaml.safe_load((HERE / 'experiment_uonly_gp_raw.yaml').read_text())
    generation = json.loads((BASE / 'config/generation.json').read_text())
    assert generation['parameters']['driver_tau_days'] == 0.5
    assert generation['parameters']['independent_gp_tau_days'] == 55.3735295468
    cfg = {
        'experiment': {'seed': generation['illumination_seed'], 'output_dir': OUT.name, 'simulation_source': 'LTSIM'},
        'observations': deepcopy(original_cfg['observations']),
        'paired_ablation': {'cases': {name: dict(blr=True, shared_driver=False, shared_response=True, unfiltered_response_gp=True) for name in names}},
        'mica_fit': deepcopy(original_cfg['mica_fit']),
        'generation': {'input_case': str(BASE), 'manifest': str(BASE / 'config/generation.json'),
                       'parameters': generation['parameters'], 'alpha': generation['alpha'],
                       'amplitude_rule': generation['normalization'], 'inference_only': True},
        'detrending': {'input_case': str(BASE), 'bands': ['U'], 'sigma_days': list(SIGMAS),
                       'formula': 'original - Gaussian weighted trend + original mean',
                       'error_mode': 'original', 'inference_only': True},
    }
    spec = importlib.util.spec_from_file_location('smooth_prepare', ROOT / 'test/mica_smooth/prepare.py')
    smooth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smooth)
    raw = np.loadtxt(BASE / 'input/LTSIM_U.txt')
    original_parts = np.loadtxt(BASE / 'components/U_components.txt')
    np.testing.assert_array_equal(raw[:, 0], original_parts[:, 0])
    np.testing.assert_allclose(raw[:, 1], 1 + original_parts[:, 1:5].sum(axis=1), atol=1e-13)
    prepared = []
    for sigma in SIGMAS:
        filtered, covariance = smooth.detrend_curve(raw, sigma, 'original')
        weights = smooth.gaussian_weights(raw[:, 0], sigma)
        parts = original_parts.copy()
        parts[:, 1:] = original_parts[:, 1:] - weights @ original_parts[:, 1:] + original_parts[:, 1:].mean(axis=0)
        np.testing.assert_array_equal(filtered[:, [0, 2]], raw[:, [0, 2]])
        np.testing.assert_allclose(filtered[:, 1], raw[:, 1] - weights @ raw[:, 1] + raw[:, 1].mean(), atol=1e-13)
        np.testing.assert_allclose(filtered[:, 1], 1 + parts[:, 1:5].sum(axis=1), atol=1e-13)
        assert np.isfinite(filtered).all() and np.all(filtered[:, 1] > 0)
        prepared.append((sigma, filtered, covariance, weights, parts))
    for sigma, filtered, covariance, weights, parts in prepared:
        case = OUT / f'{BASE.name}_u_s{sigma}'
        for directory in ('input', 'components', 'config'):
            (case / directory).mkdir(parents=True, exist_ok=True)
        for relative in ('input/LTSIM_UVW2.txt', 'components/UVW2_components.txt', 'config/generation.json'):
            shutil.copy2(BASE / relative, case / relative)
        np.savetxt(case / 'input/LTSIM_U.txt', filtered, fmt='%.17g', header='time flux error; original - Gaussian trend + original mean')
        np.save(case / 'input/U_filtered_noise_covariance.npy', covariance)
        np.savetxt(case / 'input/U_removed_trend.txt', np.column_stack([raw[:, 0], weights @ raw[:, 1]]), fmt='%.17g', header='time Gaussian_trend')
        np.savetxt(case / 'components/U_components.txt', parts, fmt='%.17g', header=(
            'time illumination_variation shared_long_variation shared_short_variation measurement_noise disk_illumination_variation blr_illumination_variation\n'
            'All U components use the same high-pass operator; last two columns partition illumination.'))
        provenance = {'original_case': str(BASE), 'sigma_days': sigma, 'driver_unchanged': True,
                      'formula': cfg['detrending']['formula'], 'quoted_errors': 'original',
                      'full_noise_covariance': 'saved for diagnosis; native MICA uses original diagonal errors',
                      'rms_retained': {name: float(np.std(parts[:, i]) / np.std(original_parts[:, i]))
                                       for name, i in [('disk', 5), ('BLR', 6), ('independent_S', 2)]}}
        (case / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        assert (case / 'input/LTSIM_UVW2.txt').read_bytes() == (BASE / 'input/LTSIM_UVW2.txt').read_bytes()
        np.testing.assert_array_equal(np.loadtxt(case / 'input/LTSIM_U.txt')[:, [0, 2]], raw[:, [0, 2]])
        print(json.dumps({'case': case.name, **provenance}, indent=2))
    CONFIG.write_text('# Inference-only saved inputs. Prepare only with prepare_fast_driver_recovery.py; do not use generic prepare.py.\n' + yaml.safe_dump(cfg, sort_keys=False))
    (OUT / 'README_recovery.md').write_text(
        '# Matched fast-driver recovery experiment\n\n'
        'Five inputs: raw, U-only Gaussian high-pass sigma5/10/20/40 observer days. Each is fit with two Gaussian and two Gamma components, ten fits in total. '
        'All inputs derive from one saved tau_I=0.5 d, tau_S=55.3735295468 d realization; W2 is byte-identical across cases. '
        'U uses original minus normalized Gaussian trend plus original sample mean; quoted errors and priors are unchanged. '
        'The full transformed noise covariance is saved but is not consumed by native MICA.\n\n'
        'Both models retain max_num_saves=10000, ptol=0.01, two MPI ranks, one particle per rank and the previous centroid/width/amplitude priors. '
        'Run fit.py --config experiment_uonly_gp_fast_driver.yaml --suite paired_ablation --batch-size 10 --run --plot-results. '
        'Native fits run in parallel; the parent then generates transfer functions, light-curve diagnostics and corner figures serially. '
        'C0/C1 in posterior transfer/corner figures are sorted by centroid within each draw. Native reconstructed-component curves retain native labels.\n')
    print('Prepared five cases / ten fits:', CONFIG)


if __name__ == '__main__':
    main()
