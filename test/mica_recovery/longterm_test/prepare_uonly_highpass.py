"""Paired sigma20/40 U-only detrending of saved raw-GP simulation; no fitting."""
from pathlib import Path
import importlib.util
import json
import shutil
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('smooth_prepare',ROOT/'test/mica_smooth/prepare.py')
smooth=importlib.util.module_from_spec(spec);spec.loader.exec_module(smooth)
base=HERE/'cases/abl_disk_blr_uonly_gp_raw'
cfg=yaml.safe_load((HERE/'experiment_uonly_gp_raw.yaml').read_text())
cfg['experiment']['output_dir']='cases_uonly_gp_highpass'
cfg['paired_ablation']['cases']={f'uonly_gp_raw_u_s{s}':dict(blr=True,shared_driver=False,shared_response=True,unfiltered_response_gp=True) for s in (20,40)}
cfg['detrending']={'input_case':str(base),'bands':['U'],'sigma_days':[20,40],'formula':'original - Gaussian weighted trend + original mean','error_mode':'original','inference_only':True}
config_path=HERE/'experiment_uonly_gp_highpass.yaml'
if config_path.exists() or (HERE/cfg['experiment']['output_dir']).exists():
    raise FileExistsError('New experiment already exists; refusing overwrite')
for sigma in (20,40):
    case=HERE/cfg['experiment']['output_dir']/f'uonly_gp_raw_u_s{sigma}'
    (case/'input').mkdir(parents=True)
    (case/'components').mkdir()
    shutil.copy2(base/'input/LTSIM_UVW2.txt',case/'input/LTSIM_UVW2.txt')
    raw=np.loadtxt(base/'input/LTSIM_U.txt')
    filtered,cov=smooth.detrend_curve(raw,sigma,'original')
    assert np.all(filtered[:,1]>0)
    assert np.array_equal(raw[:,[0,2]],filtered[:,[0,2]])
    np.savetxt(case/'input/LTSIM_U.txt',filtered,fmt='%.12g',header='time flux error; original - Gaussian trend + original mean')
    np.save(case/'input/U_filtered_noise_covariance.npy',cov)
    np.savetxt(case/'input/U_removed_trend.txt',np.column_stack([raw[:,0],raw[:,1]-filtered[:,1]+raw[:,1].mean()]),header='time Gaussian_trend')
    shutil.copy2(base/'components/UVW2_components.txt',case/'components/UVW2_components.txt')
    parts=np.loadtxt(base/'components/U_components.txt')
    assert np.allclose(parts[:,0],raw[:,0])
    weights=smooth.gaussian_weights(raw[:,0],sigma)
    original=parts[:,1:].copy()
    parts[:,1:]=original-weights@original+original.mean(axis=0)
    assert np.allclose(1+parts[:,1:5].sum(axis=1),filtered[:,1],atol=1e-9)
    np.savetxt(case/'components/U_components.txt',parts,fmt='%.12g',header='time illumination_variation shared_long_variation shared_short_variation measurement_noise disk_illumination_variation blr_illumination_variation\nAll U columns detrended; shared_long is raw independent GP; last two columns partition illumination.')
    provenance={'original_case':str(base),'sigma_days':sigma,'driver_unchanged':True,'error_mode':'original','filter':cfg['detrending']['formula'],'rms_retained':{name:float(np.std(parts[:,i])/np.std(original[:,i-1])) for name,i in [('independent',2),('disk',5),('blr',6)]}}
    (case/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance))
config_path.write_text('# Inference-only paired detrending: do not use prepare.py to regenerate these inputs.\n'+yaml.safe_dump(cfg,sort_keys=False))
(HERE/cfg['experiment']['output_dir']/'README.md').write_text('Paired U-only Gaussian detrending (sigma20/40 observer days) of abl_disk_blr_uonly_gp_raw. W2 copied unchanged; no new random realization. Original errors retained for MICA; transformed full noise covariance saved separately. Both models retain original prior and sampling settings. components files are transformed using the same linear operator; last two columns partition illumination. Use fit.py with experiment_uonly_gp_highpass.yaml; do not run the generic prepare.py with this configuration. Original unfiltered fits are the comparison baseline.\n')
