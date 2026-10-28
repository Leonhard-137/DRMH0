#!/usr/bin/env python3
"""Read-only scientific audit of existing F9 broad U/V posteriors.
Writes only a new diagnostic report; never fits, changes priors, or selects a
preferred physical branch. All probabilities use original joint-row weights.
"""
from pathlib import Path
import os,sys,json,csv
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
from collect_mica_results import param_names,normalized_weights,weighted_quantile,reorder_component_columns,component_parameter_arrays
OUT=HERE/'outputs/uv_focus_20260905'

def ranks(x,w):
    order=np.argsort(x,kind='stable');v=x[order];ww=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(v)!=0)+1];ends=np.r_[starts[1:],len(v)]
    mass=np.add.reduceat(ww,starts);mid=np.cumsum(mass)-mass/2;r=np.empty(len(v))
    for lo,hi,m in zip(starts,ends,mid):r[order[lo:hi]]=m
    return r

def corr(x,y,w):
    a=ranks(x,w);b=ranks(y,w);a-=w@a;b-=w@b
    return float(np.sum(w*a*b)/np.sqrt(np.sum(w*a*a)*np.sum(w*b*b)))

def q(v,w):return weighted_quantile(v,w,[.15865,.5,.84135]).tolist()

def main():
    refs={r['band']:float(r['theory_relative_lag_days']) for r in csv.DictReader((ROOT/'F9_a/raw/theoretical_disk_lags.csv').open())}
    report={'result_family':'F9_a/runs/mica/gamma/broad_np4/2comp','reference_file':str(ROOT/'F9_a/raw/theoretical_disk_lags.csv'),
      'reference_note':'Fixed X=2.49 scalar lag benchmark, used unchanged from earlier displays. Theory uncertainty and disk-component identity are not established here.',
      'condition_note':'Conditional summaries describe the current joint posterior. They do not replace wider-prior fits and are not posterior cuts or discovered clusters.', 'bands':[]}
    for band in ['U','V']:
        p=ROOT/f'F9_a/runs/mica/gamma/broad_np4/2comp/run_W2_to_{band}_2comp_gamma/data';x=np.loadtxt(p/'sample1d.txt_2');wt=np.loadtxt(p/'weights.txt');info=np.loadtxt(p/'sample_info1d.txt_2')
        assert len(x)==len(wt)==len(info)==10000 and np.allclose(wt[:,1],info[:,1],rtol=1e-6,atol=1e-5)
        w=normalized_weights(wt[:,-1]);names=param_names(p/'para_names_line.txt_2');x,sw=reorder_component_columns(x,names,2,'gamma',{'component_order':'moment1'});_,width,amp,tau=component_parameter_arrays(x,names,2,'gamma');frac=amp/amp.sum(axis=0)
        r={'band':band,'source':str(p),'reference_days':refs[band],'tau_short_q68':q(tau[0],w),'tau_short_q95':weighted_quantile(tau[0],w,[.025,.975]).tolist(),
          'posterior_mass_tau_short_le_fixed_reference':float(w[tau[0]<=refs[band]].sum()),'short_centroid_lower_edge_mass':float(w[tau[0]<=-4.5].sum()),
          'short_centroid_upper_edge_mass':float(w[tau[0]>=94.5].sum()),'long_centroid_upper_edge_mass':float(w[tau[1]>=94.5].sum()),
          'weighted_rank_correlations':{name:corr(tau[0],v,w) for name,v in [('ln_width_short',np.log(width[0])),('fraction_short',frac[0]),('tau_long',tau[1]),('ln_width_long',np.log(width[1]))]},'conditions':{}}
        for name,mask in [('long_tau_le_80d',tau[1]<=80),('long_tau_upper_edge',tau[1]>=94.5),('long_area_lt_1pct',frac[1]<.01),('long_area_ge_1pct',frac[1]>=.01),('long_area_lt_5pct',frac[1]<.05)]:
            ww=w[mask];r['conditions'][name]={'posterior_mass':float(ww.sum()),'conditional_weight_ess':float(ww.sum()**2/np.sum(ww**2)),
              'tau_short_q68':q(tau[0,mask],ww),'width_short_q68':q(width[0,mask],ww),'fraction_short_q68':q(frac[0,mask],ww)}
        report['bands'].append(r)
    cadence=[]
    for band in ['W2','U','V','B']:
        runband='U' if band=='W2' else band;name='cont.txt' if band=='W2' else 'line.txt'
        p=ROOT/f'F9_a/runs/mica/gamma/broad_np4/2comp/run_W2_to_{runband}_2comp_gamma/input/{name}'
        x=np.loadtxt(p);dt=np.diff(x[:,0]);source=ROOT/f'F9_a/fflux/host_sub/F9_a_{band}.txt'
        cadence.append({'band':band,'input_file':str(p),'source_file':str(source),'input_identical_to_source':p.read_bytes()==source.read_bytes(),
          'n':len(x),'span_days':float(x[-1,0]-x[0,0]),'median_cadence_days':float(np.median(dt)),'p90_cadence_days':float(np.quantile(dt,.9)),
          'max_gap_days':float(dt.max()),'gaps_gt10days':int(np.sum(dt>10)),'gaps_gt30days':int(np.sum(dt>30)),
          'strictly_increasing':bool(np.all(dt>0)),'all_finite':bool(np.isfinite(x).all()),'positive_errors':bool(np.all(x[:,2]>0))})
    report['cadence']=cadence;OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for b in report['bands']:print(b['band'],b['tau_short_q68'],b['posterior_mass_tau_short_le_fixed_reference'],b['weighted_rank_correlations'])
    print('saved',OUT/'results.json')
if __name__=='__main__':main()
