#!/usr/bin/env python3
"""Remove only the saved BLR realization from the direct-S U experiment."""
from pathlib import Path
import json, shutil, sys
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'cases/abl_disk_blr_uonly_gp_raw'
TARGET = ROOT / 'cases/abl_disk_uonly_gp_raw'
CONFIG = ROOT / 'experiment_disk_uonly_gp_raw.yaml'

def main():
    if TARGET.exists() or CONFIG.exists():
        raise FileExistsError('Refusing to overwrite the prepared case or configuration')
    original = yaml.safe_load((ROOT/'experiment_uonly_gp_raw.yaml').read_text())
    config = yaml.safe_load((ROOT/'experiment_uonly_gp_raw.yaml').read_text())
    config['paired_ablation']['cases'] = {
        TARGET.name: {'blr': False, 'shared_driver': False,
                      'shared_response': True, 'unfiltered_response_gp': True}}
    config['saved_input_preparation'] = {
        'inference_only': True, 'source_case': str(SOURCE),
        'script': str(Path(__file__).resolve()),
        'operation': 'Copy W2; subtract saved U BLR component (column 6, zero based) from U flux.',
        'no_new_random_draw': True, 'no_rescaling': True,
        'slow_process': 'A_S*S(t), directly added to U; no disk convolution or interband shift',
        'model': {'W2': '1 + alpha*(D_W*I) + noise_W',
                  'U': '1 + alpha*a*(D_U*I) + A_S*S + noise_U'}}
    curves = {b: np.loadtxt(SOURCE/'input'/f'LTSIM_{b}.txt') for b in ['UVW2','U']}
    parts = {b: np.loadtxt(SOURCE/'components'/f'{b}_components.txt') for b in ['UVW2','U']}
    for b in curves:
        np.testing.assert_array_equal(curves[b][:,0], parts[b][:,0])
        assert parts[b].shape[1] == 7
    np.testing.assert_array_equal(parts['UVW2'][:,2:4], 0)
    u = curves['U'].copy()
    u[:,1] -= parts['U'][:,6]
    np.testing.assert_allclose(u[:,1], 1 + parts['U'][:,5] + parts['U'][:,2] + parts['U'][:,3] + parts['U'][:,4], rtol=0, atol=1e-10)
    assert np.isfinite(u).all() and np.all(u[:,1:] > 0)
    for name in ['input','components','config','logs']:
        (TARGET/name).mkdir(parents=True, exist_ok=False)
    shutil.copy2(SOURCE/'input/LTSIM_UVW2.txt', TARGET/'input/LTSIM_UVW2.txt')
    np.savetxt(TARGET/'input/LTSIM_U.txt', u, fmt='%.17g')
    for band in ['UVW2','U']:
        out = parts[band].copy()
        out[:,1] = out[:,5]
        out[:,6] = 0
        header = (
            'time illumination_variation shared_long_variation shared_short_variation measurement_noise disk_illumination_variation blr_illumination_variation\n'
            f'band={band}; time_zero_mjd=59176.004915; flux=1+columns[1:5].sum; last two columns partition illumination\n'
            'model: W2=1+alpha*D_W*I+noise_W; U=1+alpha*a*D_U*I+A_S*S+noise_U; S independent of I; no BLR\n'
            'driver_tau_days=30; independent_gp_tau_days=55.3735295468; independent_gp_amplitude=0.15396862368414638; alpha=0.15291719122700673; response_disk_gain=0.787; response_blr_gain=0; blr=0; shared_driver=0; shared_response=1; unfiltered_response_gp=1\n'
            't1_kelvin=5900; inclination_deg=45; redshift=0.031; driver_wavelength_observed_angstrom=1928; response_wavelength_observed_angstrom=3465; observer_frame=true\n'
            f'source_case={SOURCE}; saved disk, S, epochs and noise retained; no random draws, filtering or per-case rescaling\n'
            'S seed=2026091901 spawn_key=(0,1); disk/BLR conditional decomposition seed=2026091902; all reused from source'
        )
        np.savetxt(TARGET/'components'/f'{band}_components.txt', out, fmt='%.17g', header=header)
    CONFIG.write_text('# Inference-only paired input: remove the saved BLR, retain all other realizations.\n' + yaml.safe_dump(config, sort_keys=False))
    checks = {}
    assert (TARGET/'input/LTSIM_UVW2.txt').read_bytes() == (SOURCE/'input/LTSIM_UVW2.txt').read_bytes()
    checks['W2_input_byte_identical'] = True
    new = np.loadtxt(TARGET/'input/LTSIM_U.txt')
    np.testing.assert_array_equal(new[:,[0,2]], curves['U'][:,[0,2]])
    np.testing.assert_allclose(curves['U'][:,1]-new[:,1], parts['U'][:,6], rtol=0, atol=3e-16)
    saved_parts = np.loadtxt(TARGET/'components/U_components.txt')
    np.testing.assert_array_equal(saved_parts[:,[0,2,3,4,5]], parts['U'][:,[0,2,3,4,5]])
    np.testing.assert_array_equal(saved_parts[:,6], 0)
    sys.path.insert(0,str(ROOT))
    from fit import source_config
    native_original = yaml.safe_load((SOURCE/'config/source_config.yaml').read_text())
    assert source_config(config) == native_original
    assert config['mica_fit'] == original['mica_fit']
    checks.update(U_times_errors_unchanged=True, disk_S_noise_reused_exactly=True,
                  source_minus_new_equals_saved_BLR=True, same_MICA_configuration=True,
                  U_flux_range=[float(new[:,1].min()),float(new[:,1].max())],
                  source_case=str(SOURCE), target_case=str(TARGET), config=str(CONFIG))
    (TARGET/'config/input_validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__ == '__main__':
    main()
