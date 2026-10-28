"""Prepare paired sigma5/10 U-only detrending from the existing saved realization.

This extends the sigma20/40 experiment without changing its inputs or fits.
Uses the same smoothing implementation and MICA configuration.
"""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import shutil
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIGMAS = (5, 10)
CONFIG = HERE / 'experiment_uonly_gp_highpass_s5_s10.yaml'
OUT = HERE / 'cases_uonly_gp_highpass_s5_s10'


def main():
    if CONFIG.exists() or OUT.exists():
        raise FileExistsError('sigma5/10 outputs already exist; refusing to overwrite')
    cfg = deepcopy(yaml.safe_load((HERE/'experiment_uonly_gp_highpass.yaml').read_text()))
    base = Path(cfg['detrending']['input_case'])
    cfg['experiment']['output_dir'] = OUT.name
    switches = dict(blr=True, shared_driver=False, shared_response=True, unfiltered_response_gp=True)
    cfg['paired_ablation']['cases'] = {f'uonly_gp_raw_u_s{s}':deepcopy(switches) for s in SIGMAS}
    cfg['detrending']['sigma_days'] = list(SIGMAS)
    cfg['detrending']['bands'] = ['U']
    assert cfg['detrending']['inference_only'] is True
    assert cfg['detrending']['error_mode'] == 'original'
    spec = importlib.util.spec_from_file_location('smooth_prepare', ROOT/'test/mica_smooth/prepare.py')
    smooth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smooth)
    raw = np.loadtxt(base/'input/LTSIM_U.txt')
    original_parts = np.loadtxt(base/'components/U_components.txt')
    assert np.array_equal(raw[:,0], original_parts[:,0])
    assert np.allclose(1+original_parts[:,1:5].sum(axis=1),raw[:,1],atol=1e-9,rtol=0)
    for sigma in SIGMAS:
        case = OUT/f'uonly_gp_raw_u_s{sigma}'
        (case/'input').mkdir(parents=True)
        (case/'components').mkdir()
        shutil.copy2(base/'input/LTSIM_UVW2.txt',case/'input/LTSIM_UVW2.txt')
        shutil.copy2(base/'components/UVW2_components.txt',case/'components/UVW2_components.txt')
        filtered, covariance = smooth.detrend_curve(raw,sigma,'original')
        assert np.all(filtered[:,1]>0) and np.isfinite(filtered).all()
        assert np.array_equal(raw[:,[0,2]],filtered[:,[0,2]])
        weights = smooth.gaussian_weights(raw[:,0],sigma)
        assert np.allclose(filtered[:,1],raw[:,1]-weights@raw[:,1]+raw[:,1].mean(),atol=1e-13,rtol=0)
        np.savetxt(case/'input/LTSIM_U.txt',filtered,fmt='%.12g',header='time flux error; original - Gaussian trend + original mean')
        np.save(case/'input/U_filtered_noise_covariance.npy',covariance)
        np.savetxt(case/'input/U_removed_trend.txt',np.column_stack([raw[:,0],weights@raw[:,1]]),header='time Gaussian_trend')
        parts = original_parts.copy()
        original = original_parts[:,1:]
        parts[:,1:] = original - weights@original + original.mean(axis=0)
        assert np.allclose(1+parts[:,1:5].sum(axis=1),filtered[:,1],atol=1e-9,rtol=0)
        np.savetxt(case/'components/U_components.txt',parts,fmt='%.12g',header='time illumination_variation shared_long_variation shared_short_variation measurement_noise disk_illumination_variation blr_illumination_variation\nAll U columns detrended; shared_long is raw independent GP; last two columns partition illumination.')
        rms_original = {name:float(np.std(original_parts[:,i])) for name,i in [('independent',2),('disk',5),('blr',6)]}
        rms_filtered = {name:float(np.std(parts[:,i])) for name,i in [('independent',2),('disk',5),('blr',6)]}
        provenance = {
            'original_case':str(base),'sigma_days':sigma,'driver_unchanged':True,
            'error_mode':'original','filter':cfg['detrending']['formula'],
            'rms_original':rms_original,'rms_filtered':rms_filtered,
            'rms_retained':{name:rms_filtered[name]/rms_original[name] for name in rms_original},
            'independent_to_blr_rms':rms_filtered['independent']/rms_filtered['blr'],
            'sampling_and_priors_from':str(HERE/'experiment_uonly_gp_highpass.yaml'),
        }
        (case/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        print(json.dumps(provenance,indent=2))
    CONFIG.write_text('# Inference-only paired detrending. Do not regenerate with prepare.py.\n'+yaml.safe_dump(cfg,sort_keys=False))
    (OUT/'README.md').write_text('''# Paired U-only high-pass sigma5/10 experiment

Input realization: ../cases/abl_disk_blr_uonly_gp_raw. No fresh random draw.
W2 is copied unchanged. U = original - Gaussian weighted trend + original mean,
with sigma=5 or 10 observer-frame days; all component columns receive the same
linear operation. Epochs and quoted errors are retained. The full transformed
noise covariance is saved for reference; MICA keeps the original diagonal errors.

Configuration: ../experiment_uonly_gp_highpass_s5_s10.yaml (inference only).
Prepare reproducibly with ../prepare_uonly_highpass_s5_s10.py, which refuses
to overwrite existing outputs. Do not use the generic prepare.py for this config.
Sampling and priors match the sigma20/40 control: two components, positive
amplitudes, centroids [0,100] d, max_num_saves=10000, ptol=0.01, two MPI ranks
per fit, one particle per rank. Gaussian sigma is [0.14195358,152.84] d;
Gamma physical standard deviation is [0.20075267806341413,216.14840087310387] d.

run_pipeline.sh launches all four case/model fits concurrently through the
existing fit.py --batch-size 4 runner, then serially produces transfer functions,
light-curve diagnostics and corner plots. Transfer-function and corner component
labels are ordered by centroid within each posterior draw: C0 <= C1. Saved
native component light curves retain their explicit Native 0/1 labels.

Results live in each case's results/{gamma,gaussian}/. Detailed native per-fit
logs and run_mica_state.json are in each case's runs directory. pipeline.log
records the outer scheduler; pipeline_exit_code records the final command status.
''')
    print('Config:',CONFIG)


if __name__ == '__main__':
    main()
