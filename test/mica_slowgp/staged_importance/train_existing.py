"""Train frozen proposals from completed child posterior files; never call GP."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.special import logsumexp,ndtr

from importance import conditional_prior_coordinates,logit_open,fit_gaussian_mixture,freeze_student_mixture,FrozenProposal

ROOT=Path('/home/leonhard/sci001/output/mica_slowgp/stability_v3_stratified/dynesty/shared_shared/budget_400')
OUT=Path(__file__).resolve().parent/'artifacts/trained'
SEEDS=(71231,71247,71263)


if __name__=='__main__':
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    reports=[]
    for layer in range(3):
        sources=[];rows=[];weights=[];configs=[]
        for seed in SEEDS:
            folder=ROOT/f'seed_{seed}/strata/stratum_{layer:02d}'
            for filename in ('sampler_config.json','nested_results.npz','diagnostics.json','posterior_summary.json','weighted_posterior.csv'):
                if not (folder/filename).is_file():raise ValueError(f'Child is not complete: {folder}')
            config=json.loads((folder/'sampler_config.json').read_text());configs.append(config)
            with np.load(folder/'nested_results.npz',allow_pickle=False) as raw:
                x=raw['samples'].copy();logw=raw['logwt'].copy();logz=float(raw['logz'][-1])
            if not np.isclose(logsumexp(logw),logz,rtol=0,atol=1e-8):raise ValueError('Child log weights/evidence mismatch')
            w=np.exp(logw-logsumexp(logw))/len(SEEDS)
            rows.append(x);weights.append(w)
            sources.append({'seed':seed,'path':str(folder/'nested_results.npz'),
                'sha256':hashlib.sha256((folder/'nested_results.npz').read_bytes()).hexdigest(),
                'rows':len(x),'conditional_log_evidence':logz,'normalized_training_share':1/len(SEEDS)})
        config=configs[0];names=config['parameter_names'];priors=config['priors'];interval=config['center_interval']
        if any(c['parameter_names']!=names or c['priors']!=priors or c['center_interval']!=interval or c['mode']!=config['mode'] for c in configs):raise ValueError('Incompatible training priors or modes')
        x=np.concatenate(rows);w=np.concatenate(weights)
        u=conditional_prior_coordinates(x,names,priors,interval)
        positive=w>0
        if not np.isfinite(u[positive]).all() or np.any(u[positive]<=0) or np.any(u[positive]>=1):raise ValueError('Positive-weight training row cannot be represented in open prior coordinates; no clipping or silent removal')
        # Exactly zero-weight source rows carry no weighted training objective.
        x=x[positive];u=u[positive];w=w[positive];z=logit_open(u)
        a0=x[:,names.index('A0')];a1=x[:,names.index('A1')]
        mu0=x[:,names.index('mu0')];mu1=x[:,names.index('mu1')]
        width0=x[:,names.index('w0')];width1=x[:,names.index('w1')]
        negative=(a0*ndtr(-mu0/width0)+a1*ndtr(-mu1/width1))/(a0+a1)
        tail_weights=w*negative
        if tail_weights.sum()<=0:raise ValueError('No represented negative-lag tail mass to train')
        train_seed=860100+100*layer
        print(f'training stratum {layer}: {len(z)} positive-weight rows',flush=True)
        base=fit_gaussian_mixture(z,w,n_components=6,seed=train_seed,max_iter=60)
        tail=fit_gaussian_mixture(z,tail_weights,n_components=6,seed=train_seed+1,max_iter=60)
        metadata={'purpose':'candidate importance proposal training only; no fresh science evaluation',
            'stratum_id':layer,'center_interval':interval,'stratum_prior_mass':config['center_prior_mass'],
            'parameter_names':names,'priors':priors,'mode':config['mode'],
            'coordinate_mapping':'current conditional sorted-mu1 inverse CDF; no A/w permutation',
            'sources':sources,'source_code_reference':'/home/leonhard/software/MICA2-slowgp/archive/stability-v3-stratified-fix1',
            'seed_mixture':'equal mixture of each seed normalized conditional posterior; no evidence reweighting across seeds',
            'base_training_weights':'w_post','tail_training_weights':'w_post*f_minus',
            'f_minus':'[A0*Phi(-mu0/w0)+A1*Phi(-mu1/w1)]/(A0+A1)',
            'training_tail_mean':float(w@negative/w.sum()),'zero_weight_training_rows_removed':int((~positive).sum()),
            'gp_likelihood_calls':0}
        proposal=freeze_student_mixture([base,tail],[.475,.475],epsilon=.05,df=7,inflation=1.2,metadata=metadata)
        path=OUT/f'stratum_{layer:02d}.npz';proposal.save(path)
        loaded=FrozenProposal.load(path)
        probe=z[np.linspace(0,len(z)-1,100).astype(int)]
        if not np.array_equal(proposal.logpdf_z(probe),loaded.logpdf_z(probe)):raise AssertionError('Frozen proposal roundtrip changed density')
        report={'stratum_id':layer,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'center_interval':interval,'stratum_prior_mass':config['center_prior_mass'],
            'training_rows':len(z),'zero_weight_rows_removed':int((~positive).sum()),
            'base_training_ess':base['metadata']['training_weight_ess'],
            'tail_training_ess':tail['metadata']['training_weight_ess'],
            'training_tail_mean':metadata['training_tail_mean'],
            'base_em_iterations':base['metadata']['iterations'],'tail_em_iterations':tail['metadata']['iterations'],
            'base_elapsed_seconds':base['metadata']['elapsed_seconds'],'tail_elapsed_seconds':tail['metadata']['elapsed_seconds'],
            'training_seeds':[train_seed,train_seed+1],'full_block_weights':[.475,.475],
            'defensive_prior_probability':.05,'student_df':7,'student_covariance_sd_inflation':1.2,
            'total_student_components':len(proposal.means),'sources':sources,'roundtrip_logpdf_exact':True}
        reports.append(report);print(json.dumps({k:report[k] for k in ('stratum_id','path','training_rows','base_training_ess','tail_training_ess','base_elapsed_seconds','tail_elapsed_seconds')}),flush=True)
    result={'completed':True,'strata':reports,'elapsed_seconds':time.perf_counter()-started,
            'gp_likelihood_calls':0,'scientific_fits_started':0,'fresh_science_evaluation_draws':0,
            'proposal_only':'Training outputs do not estimate evidence or establish convergence.'}
    (OUT/'training_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'elapsed_seconds':result['elapsed_seconds'],'proposal_paths':[r['path'] for r in reports]},indent=2))
