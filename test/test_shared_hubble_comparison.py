"""Physics/likelihood regression checks for the common-input four-case model."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad, trapezoid
from scipy.special import ndtr
from scipy.stats import norm

from src.shared_sed_hubble.h0 import (
    C_KMS, _prepare, _redshift_distribution, _settings, distance_scale,
    fit_case, run_comparison, source_log_likelihood, validate_inputs, validate_saved_comparison,
)


def source(**overrides):
    row = dict(source='fixture', z_saved=.004, D_mean=22., D_std=5.,
               z_cmb=.0045, z_cmb_error=0., v_flow_kms=120., sigma_v_kms=250.)
    return {**row, **overrides}


class ComparisonLikelihoodTests(unittest.TestCase):
    def test_normalized_no_velocity_likelihood(self):
        row = source(sigma_v_kms=0., z_cmb_error=0.)
        h, f = 65., .4
        mu = distance_scale(row['z_cmb'])/h
        expected = norm.logpdf(row['D_mean'],mu,np.hypot(row['D_std'],f*mu))
        self.assertAlmostEqual(float(source_log_likelihood(h,f,row)),expected,places=12)

    def test_positive_velocity_reduces_redshift_and_zero_velocity_identity(self):
        row = source(sigma_v_kms=0.)
        off = _prepare(pd.DataFrame([row]),False,_settings())[0]
        on = _prepare(pd.DataFrame([row]),True,_settings())[0]
        self.assertLess(on['z'][0],off['z'][0])
        self.assertAlmostEqual(on['z'][0],(1+row['z_cmb'])/(1+row['v_flow_kms']/C_KMS)-1)
        row['v_flow_kms']=0.
        self.assertAlmostEqual(float(source_log_likelihood(65,.4,row,False)),
                               float(source_log_likelihood(65,.4,row,True)),places=11)

    def test_velocity_integral_against_independent_adaptive_quad(self):
        # Direct integration in standardized v, not the implemented ratio CDF.
        for vm in (-180.,180.):
            row=source(v_flow_kms=vm,z_cmb=.006,sigma_v_kms=250.)
            for h,f in ((48.,0.),(85.,.6)):
                tmax=(C_KMS*row['z_cmb']-vm)/row['sigma_v_kms']
                def integrand(t):
                    v=vm+row['sigma_v_kms']*t
                    z=(1+row['z_cmb'])/(1+v/C_KMS)-1
                    mu=distance_scale(z)/h
                    return norm.pdf(t)*norm.pdf(row['D_mean'],mu,np.hypot(row['D_std'],f*mu))
                integral=quad(integrand,-10,min(10,tmax),epsabs=1e-12,epsrel=1e-9)[0]/ndtr(tmax)
                value=np.exp(source_log_likelihood(h,f,row,True,{'velocity_nodes':128}))
                self.assertAlmostEqual(value/integral,1.,delta=5e-5)

    def test_joint_z_and_velocity_errors_against_nested_quad(self):
        # Independent nested integral over z measurement and v. This checks
        # the ratio-normal reduction with both signs of the flow prediction.
        for vm in (-100.,100.):
            row=source(z_cmb=.008,z_cmb_error=.00012,v_flow_kms=vm,sigma_v_kms=180.)
            h,f=70.,.25
            def velocity_integrand(t):
                v=vm+row['sigma_v_kms']*t
                def redshift_integrand(u):
                    zcmb=row['z_cmb']+row['z_cmb_error']*u
                    z=(1+zcmb)/(1+v/C_KMS)-1
                    if z<=0: return 0.
                    mu=distance_scale(z)/h
                    return norm.pdf(u)*norm.pdf(row['D_mean'],mu,np.hypot(row['D_std'],f*mu))
                return norm.pdf(t)*quad(redshift_integrand,-8,8,epsabs=1e-10,epsrel=1e-7)[0]
            integral=quad(velocity_integrand,-8,8,epsabs=1e-10,epsrel=1e-6)[0]
            value=np.exp(source_log_likelihood(h,f,row,True,{'velocity_nodes':128}))
            self.assertAlmostEqual(value/integral,1.,delta=5e-5)

    def test_small_low_z_fixed_f_matches_analytic_gaussian_limit(self):
        row=source(z_cmb=.01,z_cmb_error=.00002,v_flow_kms=0.,sigma_v_kms=20.,D_mean=44.,D_std=8.)
        h=70.
        dz=np.hypot(row['z_cmb_error'],(1+row['z_cmb'])*row['sigma_v_kms']/C_KMS)
        step=1e-6
        derivative=(distance_scale(row['z_cmb']+step)-distance_scale(row['z_cmb']-step))/(2*step)
        expected=norm.pdf(row['D_mean'],distance_scale(row['z_cmb'])/h,
                          np.hypot(row['D_std'],derivative*dz/h))
        value=np.exp(source_log_likelihood(h,0.,row,True,{'velocity_nodes':128}))
        self.assertAlmostEqual(value/expected,1.,delta=2e-5)

    def test_nonpositive_redshift_is_explicit_not_clipped(self):
        with self.assertRaisesRegex(ValueError,'P\\(z_cos<=0\\)'):
            _prepare(pd.DataFrame([source(z_cmb=.0005,v_flow_kms=0.)]),True,_settings())
        row=source(z_cmb=.004,v_flow_kms=0.)
        z,w,info=_redshift_distribution(row,True,_settings())
        self.assertGreater(info['nonpositive_z_probability'],0.)
        self.assertLess(info['nonpositive_z_probability'],1e-6)
        self.assertTrue(np.all(z>0))
        self.assertAlmostEqual(w.sum(),1.,places=13)

    def test_unknown_redshift_error_must_be_resolved_explicitly(self):
        with self.assertRaisesRegex(ValueError,'unknown/nonfinite'):
            validate_inputs(pd.DataFrame([source(z_cmb_error=np.nan)]))

    def test_four_branches_share_inputs_and_fixed_f_is_separate(self):
        data=pd.DataFrame([source(source='one',z_cmb=.015,z_saved=.014,D_mean=65.,D_std=9.,v_flow_kms=0.,sigma_v_kms=0.),
                           source(source='two',z_cmb=.025,z_saved=.024,D_mean=98.,D_std=12.,v_flow_kms=0.,sigma_v_kms=0.)])
        original=data.copy(deep=True)
        settings=dict(h0_min=10.,h0_max=150.,h0_step=.5,f_max=3.,f_step=.01,
                      baseline_h0_step=.5,baseline_f_step=.01,velocity_nodes=12,
                      grid_h0_tolerance=.1)
        with tempfile.TemporaryDirectory() as tmp:
            result=run_comparison(data,Path(tmp),settings=settings,validate=True)
            rows={r['case']:r for r in result['comparison']}
            self.assertEqual(rows['A']['scatter_mode'],'fixed_zero')
            self.assertEqual(rows['A']['scatter_q50'],0.)
            self.assertEqual(rows['A']['h0_q50'],rows['C']['h0_q50'])
            self.assertEqual(rows['B']['h0_q50'],rows['D']['h0_q50'])
            self.assertFalse((Path(tmp)/'cases/A/scatter_posterior.csv').exists())
            self.assertTrue((Path(tmp)/'cases/B/joint_posterior.npz').exists())
            self.assertEqual(result['validation']['state'],'passed')
            summary_path=Path(tmp)/'run_summary.json'
            pristine=json.loads(summary_path.read_text())
            for broken_rows in ([], pristine['comparison'][:-1], pristine['comparison']+[pristine['comparison'][0]]):
                broken={**pristine,'comparison':broken_rows}
                summary_path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError,'Incomplete or duplicate'):
                    validate_saved_comparison(tmp)
            summary_path.write_text(json.dumps(pristine))
            case_path=Path(tmp)/'cases/A/summary.json'
            case_saved=json.loads(case_path.read_text())
            broken=json.loads(case_path.read_text());broken['summary']['h0_q50']+=1.
            case_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,'summary differs'):
                validate_saved_comparison(tmp)
            broken=json.loads(json.dumps(case_saved));broken['summary']['mle_success']=False
            case_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,'optimization did not succeed'):
                validate_saved_comparison(tmp)
            case_path.write_text(json.dumps(case_saved))
        pd.testing.assert_frame_equal(data,original)

    def test_saved_z_baseline_disables_new_measurement_error(self):
        data=pd.DataFrame([source(sigma_v_kms=0.,z_cmb_error=.0002)])
        settings=dict(h0_min=10.,h0_max=150.,h0_step=1.,f_max=3.,f_step=.03,
                      baseline_h0_step=1.,baseline_f_step=.03)
        result=fit_case(data,'baseline_saved_z',settings)
        self.assertEqual(result['fit_quality'].iloc[0].z_q50,data.iloc[0].z_saved)
        self.assertEqual(result['fit_quality'].iloc[0].z_error_used,0.)
        self.assertTrue(result['summary']['baseline_ignores_redshift_measurement_error'])


if __name__=='__main__':
    unittest.main()
