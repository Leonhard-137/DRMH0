#!/usr/bin/env python3
"""Generate approved tau=2, area-ratio=9 experiment and BOTH-band detrended cases.
No fits are launched. Existing experiment outputs are never overwritten.
"""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import numpy as np
import yaml
from prepare import _fft_covariance, _real_disk_transform, _sample_psd_covariance

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'cases_tau2_area9'
CONFIG=HERE/'experiment_tau2_area9.yaml'
REFERENCE=HERE/'cases/abl_disk_blr_uonly_gp_raw'
TAU=2.; ALPHA=1.45; SEED=2026100102
BANDS=('UVW2','U'); SIGMAS=(5,10,20,40)
SCHEMA='time illumination_variation shared_long_variation shared_short_variation measurement_noise disk_illumination_variation blr_illumination_variation'

def main():
    if OUT.exists() or CONFIG.exists():
        raise FileExistsError('Preserving existing experiment')
    reference=json.loads((REFERENCE/'config/paired_generation.json').read_text())
    p=deepcopy(reference['parameters'])
    p['driver_tau_days']=TAU
    p['response_disk_gain']=.787
    p['response_blr_gain']=.787/9
    raw={b:np.loadtxt(REFERENCE/f'input/LTSIM_{b}.txt') for b in BANDS}
    parts={b:np.loadtxt(REFERENCE/f'components/{b}_components.txt') for b in BANDS}
    for b in BANDS:
        np.testing.assert_array_equal(raw[b][:,0],parts[b][:,0])
        np.testing.assert_allclose(raw[b][:,1],1+parts[b][:,1:5].sum(axis=1),atol=1e-10)
    slow_scale=float(1.3/np.std(parts['U'][:,2]))
    p['independent_gp_amplitude']*=slow_scale
    parts['U'][:,2]*=slow_scale
    assert np.all(parts['UVW2'][:,2]==0)
    dt,n=p['covariance_dt_days'],p['covariance_n']
    omega=2*np.pi*np.fft.rfftfreq(n,dt)
    fw,aw=_real_disk_transform(p,p['driver_wavelength_observed_angstrom'])
    fu,au=_real_disk_transform(p,p['response_wavelength_observed_angstrom'])
    fb=np.exp(-1j*omega*p['blr_onset_days'])/(1+1j*omega*p['blr_theta_days'])**2
    filters=[fw,p['response_disk_gain']*fu,p['response_blr_gain']*fb]
    times=[raw['UVW2'][:,0],raw['U'][:,0],raw['U'][:,0]]
    spectrum=2*TAU/(1+(omega*TAU)**2)
    blocks=[[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(i,3):
            lag=times[j][None,:]-times[i][:,None]
            block=ALPHA**2*_fft_covariance(spectrum*filters[j]*np.conj(filters[i]),lag,dt,n)
            blocks[i][j]=block; blocks[j][i]=block.T
    covariance=np.block(blocks)
    draw,eig=_sample_psd_covariance((covariance+covariance.T)/2,np.random.default_rng(SEED))
    nw,nu=len(times[0]),len(times[1])
    wd,ud,ub=np.split(draw,[nw,nw+nu])
    for c in (wd,ud,ub): c-=c.mean()
    parts['UVW2'][:,1]=wd; parts['UVW2'][:,5]=wd; parts['UVW2'][:,6]=0
    parts['U'][:,1]=ud+ub; parts['U'][:,5]=ud; parts['U'][:,6]=ub
    spec=importlib.util.spec_from_file_location('smooth_prepare',ROOT/'test/mica_smooth/prepare.py')
    smooth=importlib.util.module_from_spec(spec); spec.loader.exec_module(smooth)
    prepared={'raw':{b:parts[b].copy() for b in BANDS}}
    weights={}
    for s in SIGMAS:
        name=f'all_s{s}'; prepared[name]={}; weights[name]={}
        for b in BANDS:
            w=smooth.gaussian_weights(raw[b][:,0],s); weights[name][b]=w
            c=parts[b].copy()
            c[:,1:]=parts[b][:,1:]-w@parts[b][:,1:]+parts[b][:,1:].mean(axis=0)
            prepared[name][b]=c
    # One constant for both bands and all cases, large enough for positive flux.
    minimum=min(c[:,1:5].sum(axis=1).min() for byband in prepared.values() for c in byband.values())
    baseline=float(max(1,np.ceil(1-minimum)))
    stats={b:{'realized_rms':{name:float(np.std(parts[b][:,col])) for name,col in [('disk',5),('BLR',6),('independent_S',2),('noise',4),('disk_plus_BLR',1)]}} for b in BANDS}
    assert stats['U']['realized_rms']['independent_S']>stats['U']['realized_rms']['disk']>stats['U']['realized_rms']['BLR']>stats['U']['realized_rms']['noise'],stats
    metadata=dict(parameters=p,alpha=ALPHA,illumination_seed=SEED,baseline_flux=baseline,
        reference_case=str(REFERENCE),slow_sample_rms_target=1.3,slow_scale=slow_scale,
        normalization='Unit stationary RMS latent DRW; fixed alpha=1.45; no disk/BLR sample rescaling. Preserved slow realization scaled to sample RMS 1.3. Common additive baseline ensures positive flux.',
        method='Joint Gaussian draw of W2 disk, U disk and U BLR driven by the same latent DRW using Fourier covariance.',
        model='W2=baseline+alpha*DW*I+noise; U=baseline+alpha*(0.787*DU*I+(0.787/9)*B*I)+S+noise; S is independent and directly added only to U.',
        disk_kernel_areas={'UVW2':aw,'U':au},minimum_covariance_eigenvalue=eig,statistics=stats)
    base_inputs={b:np.column_stack([raw[b][:,0],baseline+parts[b][:,1:5].sum(axis=1),raw[b][:,2]]) for b in BANDS}
    for name,byband in prepared.items():
        case=OUT/name
        for directory in ['input','components','config','results']: (case/directory).mkdir(parents=True,exist_ok=True)
        provenance=dict(original_case=str(OUT/'raw'),sigma_days=None if name=='raw' else int(name[5:]),driver_unchanged=name=='raw',bands=list(BANDS),baseline_flux=baseline,
            formula='original - Gaussian weighted trend + original mean' if name!='raw' else 'raw',quoted_errors='original',rms={})
        for b,c in byband.items():
            data=np.column_stack([raw[b][:,0],baseline+c[:,1:5].sum(axis=1),raw[b][:,2]])
            np.testing.assert_allclose(c[:,1],c[:,5]+c[:,6],atol=1e-12)
            np.testing.assert_array_equal(data[:,[0,2]],raw[b][:,[0,2]])
            if name!='raw':
                filtered,noise_cov=smooth.detrend_curve(base_inputs[b],provenance['sigma_days'],'original')
                np.testing.assert_allclose(data,filtered,atol=1e-12)
                np.save(case/f'input/{b}_filtered_noise_covariance.npy',noise_cov)
                np.savetxt(case/f'input/{b}_removed_trend.txt',np.column_stack([data[:,0],weights[name][b]@base_inputs[b][:,1]]),fmt='%.17g',header='time Gaussian_trend')
            assert np.isfinite(data).all() and np.all(data[:,1]>0)
            np.savetxt(case/f'input/LTSIM_{b}.txt',data,fmt='%.17g',header='time flux error')
            np.savetxt(case/f'components/{b}_components.txt',c,fmt='%.17g',header=SCHEMA+'\nflux=baseline_flux+sum(columns[1:5]); last two columns partition illumination')
            provenance['rms'][b]={k:float(np.std(c[:,col])) for k,col in [('disk',5),('BLR',6),('independent_S',2),('noise',4)]}
        (case/'config/generation.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (case/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cfg=yaml.safe_load((HERE/'experiment_uonly_gp_fast_driver.yaml').read_text())
    cfg['experiment'].update(seed=SEED,output_dir=OUT.name)
    cfg['observations']['amplitude_calibration']='fixed_alpha'
    cfg['observations'].pop('driver_window_rms',None)
    cfg['paired_ablation']['cases']={name:dict(blr=True,shared_driver=False,shared_response=True,unfiltered_response_gp=True) for name in prepared}
    cfg['generation']=dict(input_case=str(OUT/'raw'),manifest=str(OUT/'raw/config/generation.json'),parameters=p,alpha=ALPHA,baseline_flux=baseline,amplitude_rule=metadata['normalization'],inference_only=True)
    cfg['detrending']=dict(input_case=str(OUT/'raw'),bands=list(BANDS),sigma_days=list(SIGMAS),formula='original - Gaussian weighted trend + original mean',error_mode='original',inference_only=True)
    CONFIG.write_text('# Inference-only saved inputs; prepared by prepare_tau2_area9_recovery.py.\n'+yaml.safe_dump(cfg,sort_keys=False))
    (OUT/'README.md').write_text('# Tau2, disk:BLR area 9:1 recovery\n\n'+metadata['model']+'\n\n'+metadata['normalization']+f' Baseline={baseline}.\n\nFive matched inputs: raw and BOTH-band sigma5/10/20/40 detrending. Each is fit with two Gaussians and two Gammas (10 parallel fits, 2 MPI ranks each). Priors and sampling retained: max_num_saves=10000, ptol=0.01. Full filtered noise covariance saved but native likelihood retains original diagonal errors.\n\nTransfer functions, corner, and light-curve diagnostic plots are generated serially after fitting. TF/corner C0 and C1 are ordered by centroid per posterior draw.\n')
    print(json.dumps(dict(output=str(OUT),baseline=baseline,statistics=stats),indent=2))

if __name__=='__main__': main()
