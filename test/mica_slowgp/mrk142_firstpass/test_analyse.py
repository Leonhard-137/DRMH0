#!/usr/bin/env python3
"""Small deterministic tests only; no GP likelihood or sampler is called."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil

import numpy as np

import analyse as a


class AnalysisTests(unittest.TestCase):
    def test_weighted_quantiles_and_scaling(self):
        q=a.quantile(np.array([0.,1.,2.]),np.array([0.,1.,0.]),[.025,.5,.975])
        np.testing.assert_array_equal(q,[1.,1.,1.])
        p=np.array([[.2,10.,.4,.6,1e-100,.1,8.,2.,.1,.2,50.]])
        d=a.physical_derived(p,8.,3.)
        self.assertAlmostEqual(d['short_gain_raw'][0],.15)
        self.assertAlmostEqual(d['short_rms_raw'][0],.24)
        p[0,4]=.5
        d=a.physical_derived(p,8.,3.)
        self.assertGreater(d['short_rms_raw'][0],0)
        self.assertLess(d['short_rms_raw'][0],.24)

    def test_tau0_known_relation_and_no_clipping(self):
        offsets=np.linspace(-.04,.04,2001)
        weights=np.ones(len(offsets))/len(offsets)
        posts={b:(1.2*((a.WAVES[b]/1928.)**(4/3)-1)+offsets,weights) for b in a.BANDS}
        result=a.tau0_posterior(posts,10000,781)
        self.assertTrue(result['full_five_band'])
        self.assertLess(abs(result['obs']['mean']-1.2),.001)
        np.testing.assert_array_equal(result['samples_rest'],result['samples_obs']/1.0446)
        self.assertTrue(a.compare_tau0(result,result)['passed'])
        x=np.array([(a.WAVES[b]/1928.)**(4/3)-1 for b in a.BANDS])
        # Equal lag errors: least-squares projection has this independent closed form.
        expected=np.sum(result['lag_draws_obs']*x,axis=1)/np.sum(x*x)
        np.testing.assert_allclose(result['samples_obs'],expected,rtol=1e-14)
        negative={'U':(-np.linspace(.1,.2,100),np.ones(100)/100)}
        result=a.tau0_posterior(negative,100,45)
        self.assertEqual(result['probability_tau0_nonpositive'],1.)
        self.assertFalse(result['full_five_band'])

    def test_repeat_rms_pooled_width_and_log_units(self):
        x=np.linspace(-1,1,1001);w=np.ones(len(x))/len(x)
        first={'weights':w,'derived':{'mu0':x+3,'short_gain_raw':np.exp(x),'short_rms_raw':np.exp(x)}}
        second={'weights':w,'derived':{'mu0':2*x+3,'short_gain_raw':np.exp(2*x),'short_rms_raw':np.exp(2*x)}}
        result=a.compare_u(first,second)
        s1=(a.quantile(x,w,[.16,.84])[1]-a.quantile(x,w,[.16,.84])[0])/2
        self.assertAlmostEqual(result['checks']['mu0']['pooled_half68'],s1*np.sqrt(2.5))
        self.assertTrue(result['passed'])
        second['derived']['mu0']+=10
        self.assertFalse(a.compare_u(first,second)['passed'])
        second['derived']['mu0']-=10
        before=a.compare_u(first,second)
        for sample in (first,second):
            for key in ('short_gain_raw','short_rms_raw'):sample['derived'][key]*=3000
        after=a.compare_u(first,second)
        self.assertEqual(before['passed'],after['passed'])
        for key in ('short_gain_raw','short_rms_raw'):
            np.testing.assert_allclose(before['checks'][key]['absolute_displacements'],after['checks'][key]['absolute_displacements'],atol=2e-15)

    def make_completed(self,root):
        folder=root/'runs'/'U_primary_14203';folder.mkdir(parents=True)
        from mica_slowgp.sampler import default_priors
        from mica_slowgp.stratified import merge_children
        names=list(a.parameter_names('shared'));edges=[0.,5.,20.,100.]
        config=dict(mode='shared',sampler='dynesty',integration_scheme='ordered_mu1_stratified_v1',
            stratum_prior_masses=[.0025,.0375,.96],child_seeds=[1,2,3],priors=default_priors('shared'),
            parameter_names=names,seed=14203,budget={},center_strata=edges)
        (folder/'sampler_config.json').write_text(json.dumps(config))
        (folder/'completed.json').write_text(json.dumps(dict(integration_scheme='ordered_mu1_stratified_v1',number_of_strata=3)))
        (folder/'posterior_summary.json').write_text('{}')
        n=40;children=[]
        for i in range(3):
            child=folder/'strata'/f'stratum_{i:02d}';child.mkdir(parents=True)
            cc={**config,'integration_scheme':'unstratified','seed':i+1,'center_interval':edges[i:i+2]}
            (child/'sampler_config.json').write_text(json.dumps(cc))
            (child/'dynesty.checkpoint').write_bytes(b'toy placeholder; never restored')
            (child/'posterior_summary.json').write_text('{}')
            p=np.tile([.2,10,.4,.6,.5,.1,[3,10,40][i],2,.1,.2,50.],(n,1))
            p[:,3]+=np.linspace(-.1,.1,n);p[:,2]+=np.linspace(-.03,.03,n)
            w=np.arange(1,n+1,dtype=float);w/=w.sum();lw=np.log(w);lv=-np.ones(n);ll=np.zeros(n)
            raw=dict(samples=p,logl=ll,logwt=lw,logvol=lv,logz=np.zeros(n),logzerr=np.zeros(n),ncall=np.ones(n,dtype=int))
            children.append(raw);np.savez_compressed(child/'nested_results.npz',**raw)
            header=names+['log_likelihood','weight','log_prior_mass']
            np.savetxt(child/'weighted_posterior.csv',np.column_stack((p,ll,w,lv)),delimiter=',',header=','.join(header),comments='')
            (child/'diagnostics.json').write_text(json.dumps(dict(raw_log_evidence=0,raw_sample_count=n)))
        merged=merge_children(children,config['stratum_prior_masses'])
        np.savez_compressed(folder/'stratified_results.npz',**merged)
        header=names+['log_likelihood','weight','conditional_log_prior_mass','stratum_id']
        np.savetxt(folder/'weighted_posterior.csv',np.column_stack((merged['samples'],merged['logl'],merged['weights'],merged['conditional_logvol'],merged['stratum_id'])),delimiter=',',header=','.join(header),comments='')
        (folder/'diagnostics.json').write_text(json.dumps({'weight_ess':100,'max_posterior_weight':.01,'raw_sample_count':3*n}))
        prior_path=root/'priors.json';prior_path.write_text(json.dumps(config['priors']))
        data=np.column_stack((np.linspace(0,20,15),np.ones(15),np.full(15,.05)))
        for name in ('input_w2.txt','input_response.txt'):np.savetxt(folder/name,data)
        jobs=[dict(id='U_primary_14203',band='U',role='primary',seed=14203,out_dir=str(folder),priors=str(prior_path)),
              dict(id='M2_primary_14201',band='M2',role='primary',seed=14201,out_dir=str(root/'missing')),
              dict(id='U_repeat_14213',band='U',role='repeat',seed=14213,out_dir=str(root/'repeat_missing'))]
        manifest=dict(jobs=jobs,normalization={'W2':{'scale':8.},'U':{'scale':3.}},baseline_root=None,
            run_settings=dict(source=str(a.FROZEN),center_strata=edges,budget={}))
        path=root/'manifest.json';path.write_text(json.dumps(manifest))
        return folder,path,manifest

    def test_partial_parent_and_missing_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder,path,manifest=self.make_completed(root)
            before={p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            result=a.analyse(path,False,False)
            self.assertEqual(result['overall_status'],'partial')
            self.assertEqual(result['complete_primary_bands'],['U'])
            self.assertFalse(result['full_five_primary'])
            self.assertFalse(result['u_repeat']['available'])
            self.assertFalse(result['tau0']['primary']['full_five_band'])
            self.assertEqual(result['tau0']['primary']['bands'],['U'])
            after={p.relative_to(folder):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            (folder/'strata'/'stratum_01'/'nested_results.npz').unlink()
            result=a.analyse(path,False,False)
            self.assertEqual(result['overall_status'],'pending')
            self.assertEqual(result['jobs'][0]['state'],'invalid')
            self.assertEqual(result['tau0'],{})

    def test_baseline_same_estimator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            lines=['band tau weight gain width']
            for band in a.BANDS:
                for v in np.linspace(.9,1.1,101):lines.append(f'{band} {v} 1 0.5 0.2')
            (root/'mica_tf0_posterior.txt').write_text('\n'.join(lines))
            meta,posts=a.load_baseline({'baseline_root':str(root)})
            self.assertTrue(meta['available']);self.assertEqual(set(posts),set(a.BANDS))
            first=a.tau0_posterior(posts,300,12)
            second=a.tau0_posterior(posts,300,12)
            np.testing.assert_array_equal(first['samples_obs'],second['samples_obs'])

    def test_six_job_pipeline_and_reconstruction_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source,path,manifest=self.make_completed(root)
            jobs=[]
            for band,role,seed in [(b,'primary',14201+i) for i,b in enumerate(a.BANDS)]+[('U','repeat',14213)]:
                identifier=f'{band}_{role}_{seed}';folder=root/'runs'/identifier
                if folder!=source:shutil.copytree(source,folder)
                config=json.loads((folder/'sampler_config.json').read_text());config['seed']=seed
                (folder/'sampler_config.json').write_text(json.dumps(config))
                jobs.append(dict(id=identifier,band=band,role=role,seed=seed,out_dir=str(folder),priors=str(root/'priors.json')))
                manifest['normalization'][band]={'scale':3.}
            manifest['jobs']=jobs;path.write_text(json.dumps(manifest))
            before={p.relative_to(root/'runs'):p.read_bytes() for p in (root/'runs').rglob('*') if p.is_file()}
            def fake_reconstruct(folder,draws,points,seed):
                self.assertEqual((draws,points),(64,100))
                # Only a shape/protocol fixture. No conditional GP is evaluated.
                time=np.linspace(0,20,points);z=np.zeros((draws,points))
                np.savez_compressed(folder/'joint_reconstruction.npz',time=time,parameter_indices=np.zeros(draws,dtype=int),
                    offsets=np.ones((draws,2)),x=z,disk=z,blr=z,slow_w=z,slow_b=z,w=z+1,b=z+1)
                (folder/'reconstruction_metadata.json').write_text(json.dumps(dict(draws=draws,points=points,seed=seed,toy_mock=True)))
            with patch('mica_slowgp.cli.reconstruct',side_effect=fake_reconstruct) as mock:
                result=a.analyse(path,True,True)
            self.assertEqual(mock.call_count,6)
            self.assertTrue(result['all_requested_jobs_complete'])
            self.assertTrue(result['full_five_primary'])
            self.assertTrue(result['u_repeat']['passed'])
            self.assertTrue(result['tau0_u_repeat']['passed'])
            self.assertEqual(len(list((root/'analysis').glob('lightcurves_*.png'))),6)
            self.assertEqual(len(list((root/'analysis').glob('lightcurves_*.pdf'))),6)
            self.assertEqual(len(result['tau0']['primary']['bands']),5)
            self.assertEqual(len(result['tau0']['exclude_U']['bands']),4)
            after={p.relative_to(root/'runs'):p.read_bytes() for p in (root/'runs').rglob('*') if p.is_file()}
            self.assertEqual(before,after)


def render_toy(destination):
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);folder,path,manifest=AnalysisTests().make_completed(root)
        _,internal=a.load_job(manifest['jobs'][0],manifest)
        rng=np.random.default_rng(51);t=np.linspace(0,20,100)
        x=.3*np.sin(t/3)[None,:]+rng.normal(0,.02,(64,100))
        slow=.1*np.sin(t/15)[None,:]+rng.normal(0,.01,(64,100))
        arrays=dict(time=t,x=x,disk=.5*x,blr=.2*np.roll(x,5,axis=1),slow_w=slow,slow_b=.5*slow)
        arrays['w']=8+x+slow;arrays['b']=3+arrays['disk']+arrays['blr']+arrays['slow_b']
        a.plot_reconstruction(internal,arrays,destination/'toy_lightcurves')
        posts={b:(np.linspace(.1,.2,100)*((a.WAVES[b]/1928.)**(4/3)-1),np.ones(100)/100) for b in a.BANDS}
        a.plot_lags(posts,posts,a.tau0_posterior(posts,1000,17),destination/'toy_lag_wavelength')
    (destination/'README.md').write_text('These figures contain explicit deterministic toy data only. They validate rendering, not a Mrk142 fit. No GP or sampler call was used.\n')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    destination=Path(__file__).parent/'test_artifacts'
    if result.wasSuccessful():render_toy(destination)
    destination.mkdir(exist_ok=True)
    a.save_json(destination/'test_results.json',dict(tests_run=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),passed=result.wasSuccessful(),gp_calls=0,scientific_fits=0))
    raise SystemExit(0 if result.wasSuccessful() else 1)
