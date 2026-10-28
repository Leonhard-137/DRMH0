"""Train one fixed refined proposal; existing IS rows are training only."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.special import logsumexp,ndtr

from importance import (conditional_prior_coordinates,logit_open,fit_gaussian_mixture,
    freeze_student_mixture,FrozenProposal,json_safe)

NS_ROOT=Path('/home/leonhard/sci001/output/mica_slowgp/stability_v3_stratified/dynesty/shared_shared/budget_400')
PILOT=Path('/home/leonhard/sci001/output/mica_slowgp/importance_pilot_v1/seed_82103_n16384')
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts/trained'
SEEDS=(71231,71247,71263)
SETTINGS={'n_components_per_block':12,'max_iter':100,'tolerance':1e-5,
    'regularization':1e-4,'df':7,'inflation':1.2,'base_full_probability':.475,
    'tail_full_probability':.475,'defensive_prior_probability':.05,
    'NS_total_training_share':.5,'pilot_training_share':.5}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def f_minus(samples,names):
    x=np.asarray(samples)
    a0=x[:,names.index('A0')];a1=x[:,names.index('A1')]
    return (a0*ndtr(-x[:,names.index('mu0')]/x[:,names.index('w0')])+
            a1*ndtr(-x[:,names.index('mu1')]/x[:,names.index('w1')]))/(a0+a1)


if __name__=='__main__':
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    pilot_config=json.loads((PILOT/'sampler_config.json').read_text())
    pilot_diagnostics=json.loads((PILOT/'diagnostics.json').read_text())
    if not (PILOT/'completed.json').is_file():raise ValueError('Pilot not complete')
    source_hashes={str(PILOT/'sampler_config.json'):digest(PILOT/'sampler_config.json'),
                   str(PILOT/'diagnostics.json'):digest(PILOT/'diagnostics.json')}
    reports=[]
    for layer in range(3):
        sources=[];rows=[];latent=[];weights=[];configs=[]
        for seed in SEEDS:
            folder=NS_ROOT/f'seed_{seed}/strata/stratum_{layer:02d}'
            for filename in ('sampler_config.json','nested_results.npz','diagnostics.json'):
                if not (folder/filename).is_file():raise ValueError(f'Incomplete NS child: {folder}')
            config=json.loads((folder/'sampler_config.json').read_text());configs.append(config)
            names=config['parameter_names'];priors=config['priors'];interval=config['center_interval']
            path=folder/'nested_results.npz';source_hashes[str(path)]=digest(path)
            with np.load(path,allow_pickle=False) as raw:
                x=raw['samples'].copy();logw=raw['logwt'].copy();logz=float(raw['logz'][-1])
            if not np.isclose(logsumexp(logw),logz,rtol=0,atol=1e-8):raise ValueError('NS raw weights mismatch')
            w=np.exp(logw-logsumexp(logw))*(.5/len(SEEDS));keep=w>0
            u=conditional_prior_coordinates(x[keep],names,priors,interval)
            rows.append(x[keep]);latent.append(logit_open(u));weights.append(w[keep])
            sources.append({'kind':'conditional_NS','seed':seed,'path':str(path),'sha256':source_hashes[str(path)],
                'normalized_training_share':.5/len(SEEDS),'positive_weight_rows':int(keep.sum()),'zero_weight_rows_excluded':int((~keep).sum())})
        config=configs[0];names=config['parameter_names'];priors=config['priors'];interval=config['center_interval']
        if any(c['parameter_names']!=names or c['priors']!=priors or c['center_interval']!=interval or c['mode']!=config['mode'] for c in configs):raise ValueError('Incompatible NS sources')
        if pilot_config['parameter_names']!=names or pilot_config['priors']!=priors or pilot_config['mode']!=config['mode'] or pilot_config['center_strata'][layer:layer+2]!=interval:raise ValueError('Pilot target/coordinates mismatch')
        path=PILOT/f'stratum_{layer:02d}.npz';source_hashes[str(path)]=digest(path)
        with np.load(path,allow_pickle=False) as raw:
            is_x=raw['samples'].copy();is_z=raw['z'].copy();is_logw=raw['logweight'].copy()
            old_logq=raw['logq_z'].copy();identity_error=np.max(np.abs(is_logw-(raw['logl']+raw['logp_z']-raw['logq_z'])))
        if identity_error>1e-12:raise ValueError('Pilot weight identity failed')
        is_w=np.exp(is_logw-logsumexp(is_logw));keep=is_w>0
        rows.append(is_x[keep]);latent.append(is_z[keep]);weights.append(.5*is_w[keep])
        sources.append({'kind':'independent_importance_pilot','seed':pilot_config['seed'],
            'child_seed':pilot_config['child_seeds'][layer],'path':str(path),'sha256':source_hashes[str(path)],
            'normalized_training_share':.5,'positive_weight_rows':int(keep.sum()),
            'zero_weight_rows_excluded':int((~keep).sum()),'coordinate_source':'original stored z, no inverse-roundtrip clipping',
            'weight_definition':'exp(logweight-logsumexp(logweight)) within this stratum; no global stratum mass applied'})
        x=np.concatenate(rows);z=np.concatenate(latent);w=np.concatenate(weights)
        if not np.isfinite(z).all() or not np.isclose(w.sum(),1,rtol=0,atol=1e-14):raise ValueError('Invalid combined training measure')
        negative=f_minus(x,names);tail_weights=w*negative
        seed=870100+100*layer
        print(f'training refined layer {layer}: {len(z)} rows, 12+12 components, maxiter100',flush=True)
        base=fit_gaussian_mixture(z,w,n_components=12,seed=seed,max_iter=100,tolerance=1e-5,regularization=1e-4)
        tail=fit_gaussian_mixture(z,tail_weights,n_components=12,seed=seed+1,max_iter=100,tolerance=1e-5,regularization=1e-4)
        metadata={'purpose':'refined frozen importance proposal; no new likelihood evaluation',
            'stratum_id':layer,'center_interval':interval,'stratum_prior_mass':config['center_prior_mass'],
            'parameter_names':names,'priors':priors,'mode':config['mode'],
            'coordinate_mapping':'current conditional sorted-center prior; NS inverse; pilot original z; no A/w permutation',
            'sources':sources,'settings':SETTINGS,'base_training_weights':'NS seeds each1/6 plus pilot1/2',
            'tail_training_weights':'combined base training weights times f_minus, normalized by fit',
            'f_minus':'[A0*Phi(-mu0/w0)+A1*Phi(-mu1/w1)]/(A0+A1)',
            'pilot_reuse_policy':'training only; old rows are not evaluated under the new q as fresh importance samples',
            'source_code_reference':'/home/leonhard/software/MICA2-slowgp/archive/importance-pilot-v1/importance.py',
            'training_tail_mean':float(w@negative),'gp_likelihood_calls':0}
        proposal=freeze_student_mixture([base,tail],[.475,.475],epsilon=.05,df=7,inflation=1.2,metadata=metadata)
        out=OUT/f'stratum_{layer:02d}.npz';proposal.save(out);loaded=FrozenProposal.load(out)
        probe=z[np.linspace(0,len(z)-1,100).astype(int)]
        if not np.array_equal(proposal.logpdf_z(probe),loaded.logpdf_z(probe)):raise AssertionError('Frozen roundtrip density changed')
        # Training-row diagnostics only. These are NOT a new IS estimate.
        selected=np.unique(np.r_[np.argsort(is_w)[-2:],5507] if layer==1 else np.argsort(is_w)[-3:])
        logq_gain=proposal.logpdf_z(is_z[selected])-old_logq[selected]
        density_probe=[{'pilot_row':int(i),'old_conditional_posterior_weight':float(is_w[i]),
            'f_minus':float(f_minus(is_x[i:i+1],names)[0]),'new_minus_old_logq':float(gain),
            'interpretation':'in-sample proposal density placement only, not independent efficiency validation'} for i,gain in zip(selected,logq_gain)]
        report={'stratum_id':layer,'path':str(out),'sha256':digest(out),'center_interval':interval,
            'stratum_prior_mass':config['center_prior_mass'],'training_rows':len(z),
            'combined_training_max_row_weight':float(w.max()),'combined_training_tail_mean':float(w@negative),
            'settings':SETTINGS,'training_seeds':[seed,seed+1],
            'base_fit':base['metadata'],'tail_fit':tail['metadata'],'sources':sources,
            'training_row_density_probes':density_probe,'roundtrip_density_exact':True}
        reports.append(report)
        print(json.dumps({'layer':layer,'base_seconds':base['metadata']['elapsed_seconds'],'tail_seconds':tail['metadata']['elapsed_seconds'],
            'base_hit_maxiter':base['metadata']['hit_maxiter'],'tail_hit_maxiter':tail['metadata']['hit_maxiter'],
            'base_ESS':base['metadata']['training_weight_ess'],'tail_ESS':tail['metadata']['training_weight_ess']}),flush=True)
    unchanged={path:digest(Path(path))==digest_before for path,digest_before in source_hashes.items()}
    if not all(unchanged.values()):raise AssertionError('A source changed during training')
    result={'completed':True,'candidate':'refined_importance_training_v1','settings':SETTINGS,'strata':reports,
        'elapsed_seconds':time.perf_counter()-started,'source_files_unchanged':unchanged,
        'pilot_diagnostic_context':{'tail_mean':pilot_diagnostics['summaries']['total_negative_lag_fraction']['mean'],
            'tail_mcse':pilot_diagnostics['summaries']['total_negative_lag_fraction']['mean_mcse'],
            'variance_attribution_from_independent_root_and_agent_review':{'middle_stratum_fraction':.95488,'lowest_stratum_fraction':.0446,
                'middle_row_5507_global_variance_fraction':.64622,'middle_row_5507_f_minus':.0295747,
                'interpretation':'tail concentration motivates richer base and tail fits; high base weights also affect denominator uncertainty'}},
        'gp_likelihood_calls':0,'scientific_fits_started':0,'fresh_science_evaluation_draws':0,
        'old_pilot_reused_only_for_training':True,'independent_next_evaluation_required':True}
    (OUT/'training_report.json').write_text(json.dumps(json_safe(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':True,'elapsed_seconds':result['elapsed_seconds'],'paths':[r['path'] for r in reports]},indent=2),flush=True)
