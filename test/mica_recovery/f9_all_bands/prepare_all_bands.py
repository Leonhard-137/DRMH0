#!/usr/bin/env python3
"""Prepare paired W1/U/V inputs and configs; never launch MICA.

The original M2/B experiment is read only. Its epoch, physical disk settings,
driver seed and independent slow processes define this extension.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import fftconvolve, lfilter
from scipy.special import gammainc
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def independent_slow(n, dt, width, tau, seed):
    # The same PSD and RNG ordering as the original generator; no driver input.
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, dt)
    psd = 1 / ((1 + (2*np.pi*f*tau)**2) * (1 + (2*np.pi*f*width)**2)**2)
    z = rng.normal(size=len(f)) + 1j*rng.normal(size=len(f))
    z *= np.sqrt(psd/2)
    z[0] = 0
    if n % 2 == 0:
        z[-1] = rng.normal() * np.sqrt(psd[-1])
    signal = np.fft.irfft(z, n=n)
    return signal / np.std(signal)


def prepare():
    batch = yaml.safe_load((HERE/'batch_config.yaml').read_text())
    parent = (HERE/batch['parent_experiment']).resolve()
    if (HERE/'manifest.json').exists() or (HERE/'cases').exists():
        raise RuntimeError('Prepared extension already exists; refusing to regenerate it.')
    old_truth = json.loads((parent/'outputs/truth.json').read_text())
    metadata = json.loads((parent/'outputs/kernels/disk_parameters.json').read_text())
    source_cfg = yaml.safe_load((ROOT/'F9_a/config/source_config.yaml').read_text())
    z = metadata['physical_inputs_from_f9_table']['redshift']
    if source_cfg['redshift'] != z:
        raise ValueError('F9 redshift changed from the original injection.')
    for band in ('W2', *batch['reused_response_bands']):
        if source_cfg['bands'][band] != metadata['bands'][band]['wavelength_observer_angstrom']:
            raise ValueError(f'F9 wavelength changed: {band}')
    dt = old_truth['dt_days']
    n = old_truth['mother_n']
    epoch = old_truth['time_zero_mjd']
    time = old_truth['mother_start_day'] + np.arange(n)*dt
    new_bands = batch['new_response_bands']
    # Preserve children 0..5; append new noise branches, never reorder old ones.
    seeds = np.random.SeedSequence(old_truth['seed']).spawn(9)
    if batch['noise_seed_children'] != {'W1': 6, 'U': 7, 'V': 8}:
        raise ValueError('Unexpected noise seed mapping; this would change the pairing.')
    tau = old_truth['hidden_drw_tau_days']
    std = old_truth['hidden_drw_std']
    rng = np.random.default_rng(seeds[0])
    rho = np.exp(-dt/tau)
    innovation = rng.normal(0, std*np.sqrt(1-rho*rho), n)
    innovation[0] = rng.normal(0, std)
    hidden = lfilter([1], [1, -rho], innovation)
    del innovation
    sys.path.insert(0, str(parent))
    import build_disk_kernels as helper
    numerical = metadata['numerical_grid']
    geometry = metadata['simulation_assumptions']
    temperature = metadata['temperature_scaling']

    def make_kernel(band):
        disk = helper.DiskConfig(
            Rin=geometry['rin_light_days'], Rout=geometry['rout_light_days'],
            R0=geometry['r0_light_days'], TB=temperature['tb_r0_k'],
            TF=temperature['tf_r0_k'], b=0.75, inc_deg=geometry['inclination_deg'],
            lam_A=source_cfg['bands'][band]/(1+z), NR=numerical['source_nr'],
            Ntau=numerical['source_ntau'], response_mode='finite_difference')
        source_lag, density = helper.disk_transfer_function(disk)
        return helper._rebin_cdf_to_impulse_grid(source_lag*(1+z), density/(1+z), dt)

    old_w2 = np.loadtxt(parent/'outputs/kernels/disk_kernel_W2.csv', delimiter=',', skiprows=1)
    lag, w2_mass, edges, checks = make_kernel('W2')
    np.testing.assert_allclose(lag, old_w2[:,0], rtol=0, atol=1e-11)
    np.testing.assert_allclose(w2_mass, old_w2[:,3], rtol=1e-9, atol=1e-13)
    w2_curve = fftconvolve(hidden, w2_mass, mode='full')[:n]
    parent_w2 = parent/'cases/response_within/input/F9SIM_W2.txt'
    w2_input = np.loadtxt(parent_w2)
    w2_disk = np.interp(w2_input[:,0], time, w2_curve)
    w2_disk -= w2_disk.mean()
    w2_noise = np.random.default_rng(seeds[3]).normal(0, w2_input[:,2])
    w2_error = float(np.max(np.abs(1+w2_disk+w2_noise-w2_input[:,1])))
    np.testing.assert_allclose(1+w2_disk+w2_noise, w2_input[:,1], rtol=0, atol=2e-10)
    for case in batch['cases']:
        assert (parent/'cases'/case/'input/F9SIM_W2.txt').read_bytes() == parent_w2.read_bytes()

    out = HERE/'outputs'
    kernels = out/'kernels'
    kernels.mkdir(parents=True)
    curves, observed, obs_time, noise_sd, noise, disk_obs, moment = {}, {}, {}, {}, {}, {}, {}
    new_metadata = copy.deepcopy(metadata)
    new_metadata['bands'] = {}
    new_metadata['parent_metadata'] = str(parent/'outputs/kernels/disk_parameters.json')
    new_metadata['files'] = {'kernels': {}}
    w2_moment = helper._mass_moments(lag, w2_mass)
    for band in new_bands:
        band_lag, mass, band_edges, rebin = make_kernel(band)
        np.testing.assert_array_equal(band_lag, lag)
        helper._write_kernel_csv(kernels/f'disk_kernel_{band}.csv', lag, band_edges, mass)
        moment[band] = helper._mass_moments(lag, mass)
        new_metadata['bands'][band] = {
            'wavelength_observer_angstrom': source_cfg['bands'][band],
            'rebinned_moments': moment[band], 'rebin_checks': rebin,
            'relative_centroid_to_w2_days': moment[band]['mean_lag_days']-w2_moment['mean_lag_days']}
        new_metadata['files']['kernels'][band] = f'disk_kernel_{band}.csv'
        curves[band] = fftconvolve(hidden, mass, mode='full')[:n]
        path = ROOT/f'F9_a/fflux/host_sub/F9_a_{band}.txt'
        observed[band] = np.loadtxt(path)
        arr = observed[band]
        if not np.isfinite(arr).all() or np.any(np.diff(arr[:,0]) <= 0) or np.any(arr[:,2] <= 0) or arr[:,1].mean() <= 0:
            raise ValueError(f'Invalid source observations: {band}')
        obs_time[band] = arr[:,0]-epoch
        noise_sd[band] = arr[:,2]/arr[:,1].mean()
        noise[band] = np.random.default_rng(seeds[batch['noise_seed_children'][band]]).normal(0, noise_sd[band])
        sampled = np.interp(obs_time[band], time, curves[band])
        disk_obs[band] = sampled-sampled.mean()
    del hidden
    dump(kernels/'disk_parameters.json', new_metadata)
    display_min = min(0., min(t.min() for t in obs_time.values()))
    display_max = max(old_truth['baseline_days'], max(t.max() for t in obs_time.values()))
    display_idx = np.flatnonzero((time >= display_min) & (time <= display_max))[::10]
    new_truth = {
        'status': 'inputs_prepared_not_fitted', 'parent_truth_file': str(parent/'outputs/truth.json'),
        'source': 'F9_a', 'driver': 'W2', 'bands': batch['all_response_bands'],
        'new_bands': new_bands, 'seed': old_truth['seed'], 'noise_seed_children': batch['noise_seed_children'],
        'time_frame': 'observer_days', 'time_zero_mjd': epoch, 'baseline_days': old_truth['baseline_days'],
        'new_observation_extent_days': [float(display_min),float(display_max)],
        'slow_to_disk_rms_ratio_at_observed_times': old_truth['slow_to_disk_rms_ratio_at_observed_times'],
        'disk_parameters_file': str(kernels/'disk_parameters.json'),
        'relative_moment_note': old_truth['relative_moment_note'],
        'scope_note': 'Same one realization; unit-area band disk kernels and mean-one flux. No physical variable SED or U excess has been injected.',
        'cases': []}
    validation = {'w2_reconstruction_max_abs_error': w2_error, 'slow_parent_max_abs_error': {}, 'cases': {}}
    original_cases = {item['name']:item for item in old_truth['cases']}
    manifest = {'n_total_fits':25, 'n_new_fits':15, 'n_reused_fits':10, 'cases':[], 'results':[],
                'truth_file':str(out/'truth.json')}
    # Only two full slow curves are retained at a time.
    for case in batch['cases']:
        case_root = HERE/'cases'/case
        input_dir = case_root/'input'
        input_dir.mkdir(parents=True)
        (case_root/'config').mkdir()
        shutil.copyfile(parent_w2, input_dir/'F9SIM_W2.txt')
        signal = None
        width = None
        origin = 'disk_only'
        if case != 'disk_only':
            origin, scale = case.split('_',1)
            width = old_truth['slow_cases'][scale]['width_days']
            if origin == 'response':
                slow_lag = np.arange(int(np.ceil(25*width/dt))+1)*dt
                mass = gammainc(2,(slow_lag+dt/2)/width)-gammainc(2,np.maximum(slow_lag-dt/2,0)/width)
                mass /= mass.sum()
                signal = fftconvolve(w2_curve,mass,mode='full')[:n]
            else:
                signal = independent_slow(n,dt,width,tau,seeds[1 if scale=='within' else 2])
            # Check the reconstructed slow process against both saved old bands.
            for band in batch['reused_response_bands']:
                old = np.loadtxt(parent/'cases'/case/f'truth_at_{band}_observations.csv',delimiter=',',skiprows=1)
                sampled = np.interp(old[:,0],time,signal)
                sampled -= sampled.mean()
                gain = original_cases[case]['bands'][band]['slow_scale']
                error = float(np.max(np.abs(gain*sampled-old[:,2])))
                np.testing.assert_allclose(gain*sampled,old[:,2],rtol=0,atol=2e-10)
                validation['slow_parent_max_abs_error'][f'{case}/{band}'] = error
        record = {'name':case,'origin':origin,'path':str(case_root),'bands':{}}
        if case == 'disk_only':
            record['bands'].update(json.loads((parent/'cases/disk_only/truth.json').read_text())['bands'])
        else:
            record['bands'].update(copy.deepcopy(original_cases[case]['bands']))
        for band in new_bands:
            if signal is None:
                gain = 0.
                slow = np.zeros(len(obs_time[band]))
                display_slow = np.zeros(len(display_idx))
            else:
                sampled = np.interp(obs_time[band],time,signal)
                average = sampled.mean()
                sampled -= average
                gain = float(old_truth['slow_to_disk_rms_ratio_at_observed_times']*np.std(disk_obs[band])/np.std(sampled))
                if origin == 'response' and gain >= 9.5:
                    raise ValueError(f'Slow gain near native upper bound: {case}/{band}')
                slow = gain*sampled
                display_slow = gain*(signal[display_idx]-average)
            flux = 1+disk_obs[band]+slow+noise[band]
            if not np.isfinite(flux).all() or np.any(flux <= 0):
                raise ValueError(f'Invalid simulated flux: {case}/{band}')
            np.savetxt(input_dir/f'F9SIM_{band}.txt',np.column_stack((obs_time[band],flux,noise_sd[band])),fmt='%.12g')
            np.savetxt(case_root/f'truth_at_{band}_observations.csv',
                       np.column_stack((obs_time[band],disk_obs[band],slow,noise[band],flux)),
                       delimiter=',',comments='',header='observer_day,disk_variation,slow_variation,noise,observed_flux',fmt='%.12g')
            display_disk = curves[band][display_idx]-np.interp(obs_time[band],time,curves[band]).mean()
            np.savetxt(case_root/f'noiseless_{band}.csv',
                       np.column_stack((time[display_idx],display_disk,display_slow,1+display_disk+display_slow)),
                       delimiter=',',comments='',header='observer_day,disk_variation,slow_variation,total_noiseless_flux',fmt='%.10g')
            short_mu = moment[band]['mean_lag_days']-w2_moment['mean_lag_days']
            short_var = moment[band]['variance_lag_days2']-w2_moment['variance_lag_days2']
            total_mu = short_mu if signal is None else ((short_mu+gain*2*width)/(1+gain) if origin=='response' else None)
            total_var = short_var if signal is None else ((short_var+(short_mu-total_mu)**2+gain*(2*width**2+(2*width-total_mu)**2))/(1+gain) if origin=='response' else None)
            ratio = float(np.std(slow)/np.std(disk_obs[band]))
            np.testing.assert_allclose(ratio,0 if signal is None else .3,rtol=0,atol=1e-12)
            record['bands'][band] = {
                'n_points':len(flux),'slow_scale':gain,'slow_rms':float(np.std(slow)),
                'disk_rms':float(np.std(disk_obs[band])), 'short_relative_centroid_days':short_mu,
                'short_relative_second_cumulant_days2':short_var,
                'injected_slow_centroid_days':2*width if origin=='response' else None,
                'injected_slow_width_days':width if origin=='response' else None,
                'injected_total_relative_centroid_days':total_mu,
                'injected_total_relative_second_cumulant_days2':total_var,
                'observation_input':str(ROOT/f'F9_a/fflux/host_sub/F9_a_{band}.txt'),
                'input_file':str(input_dir/f'F9SIM_{band}.txt')}
            validation['cases'][f'{case}/{band}'] = {'n_points':len(flux),'slow_disk_rms_ratio':ratio}
        original_cfg = yaml.safe_load((parent/'cases'/case/'config/source_config.yaml').read_text())
        cfg = copy.deepcopy(original_cfg)
        cfg['response_bands'] = batch['all_response_bands']
        cfg['bands'] = {band:source_cfg['bands'][band] for band in ('W2',*batch['all_response_bands'])}
        cfg['mica_stages'] = [batch['stage']]
        fit = cfg[batch['stage']]
        fit['responses'] = list(new_bands)
        fit['prepared_dir'] = str(input_dir)
        cfg['simulation'] = {'case':case,'seed':old_truth['seed'],'parent_case':str(parent/'cases'/case),
                             'extension_note':'Only W1/U/V are scheduled; completed M2/B results are indexed in manifest.json.',
                             'time_zero_mjd':epoch,'noise_seed_children':batch['noise_seed_children']}
        (case_root/'config/source_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        new_truth['cases'].append(record)
        dump(case_root/'truth.json',record)
        manifest['cases'].append(str(case_root))
        for band in batch['all_response_bands']:
            reused = band in batch['reused_response_bands']
            source = parent/'cases'/case if reused else case_root
            manifest['results'].append({'case':case,'band':band,'reused':reused,
                                       'run_dir':str(source/f'runs/mica/2comp/run_W2_to_{band}_2comp_gamma')})
        del signal
    dump(out/'truth.json',new_truth)
    dump(out/'preparation_validation.json',validation)
    dump(HERE/'manifest.json',manifest)
    print('Prepared 5 cases x 3 new bands = 15 fits; 10 completed M2/B fits reused. No MICA fits started.')
    print('Injected relative disk centroids:',json.dumps({b:new_metadata['bands'][b]['relative_centroid_to_w2_days'] for b in new_bands}))


if __name__ == '__main__':
    prepare()
