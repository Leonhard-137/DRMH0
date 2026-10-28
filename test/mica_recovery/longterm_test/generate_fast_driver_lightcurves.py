#!/usr/bin/env python3
"""Generate the approved tau_I=0.5 d experiment with the saved measurement setup.

W2 = 1 + alpha D_W*I + noise_W
U  = 1 + alpha (a D_U+b B)*I + saved_S + noise_U
Joint Gaussian sampling integrates out the common DRW; it does not draw
independent disk and BLR light curves. This script never launches MICA.
"""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import _fft_covariance, _real_disk_transform, _sample_psd_covariance
from plot_style import COLORS, configure_style, observations, style_axis, save_figure

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'cases/abl_disk_blr_uonly_gp_raw'
OUTPUT = HERE / 'cases_fast_driver/disk_blr_uonly_gp_tauI0p5'
TAU_I = 0.5
SEED = 2026100101


def main():
    if OUTPUT.exists():
        raise FileExistsError(f'Preserving existing experiment: {OUTPUT}')
    reference = json.loads((REFERENCE / 'config/paired_generation.json').read_text())
    p = dict(reference['parameters'])
    p['driver_tau_days'] = TAU_I
    alpha = float(reference['alpha'])
    bands = ('UVW2', 'U')
    old = {b: np.loadtxt(REFERENCE / f'input/LTSIM_{b}.txt') for b in bands}
    parts = {b: np.loadtxt(REFERENCE / f'components/{b}_components.txt') for b in bands}
    for b in bands:
        np.testing.assert_array_equal(old[b][:, 0], parts[b][:, 0])
        np.testing.assert_allclose(old[b][:, 1], 1 + parts[b][:, 1:5].sum(axis=1), atol=1e-10)
        assert np.all(parts[b][:, 3] == 0)
    assert np.all(parts['UVW2'][:, 2] == 0)
    times = [old['UVW2'][:, 0], old['U'][:, 0], old['U'][:, 0]]
    dt, n = p['covariance_dt_days'], p['covariance_n']
    omega = 2 * np.pi * np.fft.rfftfreq(n, dt)
    fw, area_w = _real_disk_transform(p, p['driver_wavelength_observed_angstrom'])
    fu, area_u = _real_disk_transform(p, p['response_wavelength_observed_angstrom'])
    fb = np.exp(-1j * omega * p['blr_onset_days']) / (1 + 1j * omega * p['blr_theta_days'])**2
    filters = [fw, p['response_disk_gain'] * fu, p['response_blr_gain'] * fb]
    spectrum = 2 * TAU_I / (1 + (omega * TAU_I)**2)
    blocks = [[None] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            lag = times[j][None, :] - times[i][:, None]
            block = alpha**2 * _fft_covariance(
                spectrum * filters[j] * np.conj(filters[i]), lag, dt, n)
            blocks[i][j] = block
            blocks[j][i] = block.T
    covariance = np.block(blocks)
    covariance = (covariance + covariance.T) / 2
    draw, min_eig = _sample_psd_covariance(covariance, np.random.default_rng(SEED))
    nw, nu = len(times[0]), len(times[1])
    w_disk, u_disk, u_blr = np.split(draw, [nw, nw + nu])
    for curve in (w_disk, u_disk, u_blr):
        curve -= curve.mean()
    new_parts = {b: parts[b].copy() for b in bands}
    new_parts['UVW2'][:, 1] = w_disk
    new_parts['UVW2'][:, 5] = w_disk
    new_parts['UVW2'][:, 6] = 0
    new_parts['U'][:, 1] = u_disk + u_blr
    new_parts['U'][:, 5] = u_disk
    new_parts['U'][:, 6] = u_blr
    new = {}
    stats = {}
    for b in bands:
        c = new_parts[b]
        new[b] = np.column_stack([old[b][:, 0], 1 + c[:, 1:5].sum(axis=1), old[b][:, 2]])
        np.testing.assert_array_equal(c[:, [0, 2, 3, 4]], parts[b][:, [0, 2, 3, 4]])
        np.testing.assert_array_equal(new[b][:, [0, 2]], old[b][:, [0, 2]])
        np.testing.assert_allclose(c[:, 1], c[:, 5] + c[:, 6], atol=1e-15)
        assert np.isfinite(new[b]).all() and np.all(new[b][:, 1] > 0)
        stats[b] = {'n_epochs': len(c), 'span_days': float(np.ptp(c[:, 0])),
                    'median_quoted_error': float(np.median(new[b][:, 2])),
                    'realized_rms': {name: float(np.std(c[:, col])) for name, col in
                                    [('disk', 5), ('BLR', 6), ('independent_S', 2), ('noise', 4), ('disk_plus_BLR', 1)]}}
    # Scientific checks above precede output creation; old inputs are read only.
    for directory in ('input', 'components', 'config', 'results'):
        (OUTPUT / directory).mkdir(parents=True, exist_ok=True)
    for b in bands:
        np.savetxt(OUTPUT / f'input/LTSIM_{b}.txt', new[b], fmt='%.17g', header='time flux flux_error')
        np.savetxt(OUTPUT / f'components/{b}_components.txt', new_parts[b], fmt='%.17g', header=(
            'time illumination_variation shared_long_variation shared_short_variation measurement_noise disk_illumination_variation blr_illumination_variation\n'
            'flux=1+columns[1:5].sum; last two columns partition illumination; no per-realization rescaling\n'
            f'tau_I_days={TAU_I}; tau_S_days={p["independent_gp_tau_days"]}; alpha={alpha:.17g}; illumination_seed={SEED}'))
        saved = np.loadtxt(OUTPUT / f'input/LTSIM_{b}.txt')
        np.testing.assert_array_equal(saved[:, [0, 2]], old[b][:, [0, 2]])
    metadata = {'case': OUTPUT.name, 'reference_case': str(REFERENCE), 'parameters': p,
                'alpha': alpha, 'illumination_seed': SEED,
                'method': 'Joint Gaussian sampling of three responses to one latent DRW using original physical kernels and Fourier covariance integration.',
                'normalization': 'Fixed reference alpha and unit stationary variance of latent DRW. No recalibration of W2 RMS and no realization RMS rescaling.',
                'preserved': ['observed epochs', 'quoted errors', 'saved independent U slow signal', 'saved noise in both bands', 'disk kernels', 'BLR kernel', 'response gains'],
                'slow_process': 'Saved U-only Matern-3/2 GP, added directly without disk filtering or shift; no S in W2.',
                'minimum_covariance_eigenvalue': min_eig,
                'disk_kernel_areas': {'UVW2': area_w, 'U': area_u},
                'latent_DRW_theoretical_fraction_period_lt10d': 1 - 2/math.pi*math.atan(2*math.pi*TAU_I/10),
                'statistics': stats}
    (OUTPUT / 'config/generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, sharey=True)
    lower = min(float(np.min(new[b][:, 1] - new[b][:, 2])) for b in bands)
    upper = max(float(np.max(new[b][:, 1] + new[b][:, 2])) for b in bands)
    padding = (upper - lower) * .14
    for ax, b, label in zip(axes, bands, ('(a) W2', '(b) U')):
        c, values = new_parts[b], new[b]
        noiseless = values[:, 1] - c[:, 4]
        ax.plot(values[:, 0], noiseless, color=COLORS['REFERENCE'], ls=(0, (1.2, 1.8)), lw=1.05,
                label='Injected signal', zorder=3)
        if b == 'U':
            ax.plot(values[:, 0], 1 + c[:, 2], color=COLORS['SECONDARY'],
                    ls=(0, (5, 1.8, 1.3, 1.8)), lw=1.25, label=r'$1+S$', zorder=3)
        observations(ax, values[:, 0], values[:, 1], values[:, 2], label='Simulated measurements')
        ax.set_title(label, loc='left', pad=5, fontsize=12)
        ax.set_ylabel('Relative flux')
        ax.set_ylim(lower - padding, upper + padding)
        style_axis(ax, y_on_right=False)
        ax.legend(frameon=False, loc='upper right', fontsize=10, ncol=3 if b == 'U' else 2)
    axes[-1].set_xlabel('Time since MJD 59176.004915 (days)')
    axes[-1].set_xlim(min(old[b][0, 0] for b in bands) - 3, max(old[b][-1, 0] for b in bands) + 3)
    fig.subplots_adjust(left=.09, right=.98, bottom=.10, top=.95, hspace=.16)
    output_figure = OUTPUT / 'results/W2_U_light_curves'
    fig.savefig(output_figure.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    save_figure(fig, output_figure)
    (OUTPUT / 'README.md').write_text(
        '# Fast central-driver realization\n\n'
        'Generated with generate_fast_driver_lightcurves.py. Only latent DRW tau changes from 30 to 0.5 observer days; its response realization is newly drawn. '
        'The independent U slow process retains tau=55.3735295468 days and the exact saved realization. '
        'W2 has disk response only; U has disk plus BLR plus independent S. '
        'Original epochs, errors, measurement noise, physical response kernels, gains and alpha are retained. '
        'Per-band component means are removed as before; no signal RMS is rescaled.\n\n'
        'Figure: black points and gray errors are simulated measurements; blue dotted traces connect noise-free values only at observed epochs; green is 1+S. '
        'Both panels have the same flux limits. No detrending or MICA fitting is performed.\n', encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'statistics': stats, 'minimum_eigenvalue': min_eig}, indent=2))


if __name__ == '__main__':
    main()
