# 新 H0 比较核心：每个源只用冻结的 D_L 均值/标准差，不重跑 MICA/FFA。
# A/B 使用 CMB 红移，C/D 加入流场均值和剩余速度；A/C 固定 f_D=0，B/D 拟合 f_D。
# f_D 是线性距离额外相对散度，方差贡献 (f_D D_model)^2，与旧 joint_hubble 对数散度不同。
# SciPy 提供宇宙学积分、优化、正态 CDF/逆 CDF 和边缘化；所有情景使用同一批距离输入。
"""Four common-input H0 comparisons with source-level velocity marginalization.

The saved luminosity-distance mean/std and its original redshift are frozen.
The cosmology side uses K(z_cos)/H0 in every case.  This is the existing
linear-distance summary likelihood, not a full lightcurve hierarchical model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, quad, trapezoid
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri
from scipy.stats import norm

C_KMS = 299792.458
PROBS = np.array([0.02275, 0.15865, 0.5, 0.84135, 0.97725])
QUANTILE_NAMES = ('q2p3', 'q16', 'q50', 'q84', 'q97p7')
CASE_CONFIG = {
    'A': {'pv_enabled': False, 'scatter_mode': 'fixed_zero'},
    'B': {'pv_enabled': False, 'scatter_mode': 'fit'},
    'C': {'pv_enabled': True, 'scatter_mode': 'fixed_zero'},
    'D': {'pv_enabled': True, 'scatter_mode': 'fit'},
    'baseline_saved_z': {'pv_enabled': False, 'scatter_mode': 'fit'},
}
DEFAULTS = {
    'om0': 0.3, 'h0_min': 1.0, 'h0_max': 800.0, 'h0_step': 0.25,
    'f_max': 5.0, 'f_step': 0.004, 'f_prior_sigma': 1.0,
    'velocity_nodes': 64, 'redshift_error_nodes': 32, 'chunk_h0': 32,
    'max_nonpositive_z_mass': 1e-6,
    'baseline_h0_step': 0.125, 'baseline_f_step': 0.002,
    'grid_h0_tolerance': 0.05, 'grid_scatter_tolerance': 0.005,
    'quadrature_loglike_tolerance': 0.01,
}
REQUIRED = ('source', 'z_saved', 'D_mean', 'D_std', 'z_cmb',
            'z_cmb_error', 'v_flow_kms', 'sigma_v_kms')


# 合并已知设置并检查网格、节点数与先验范围；未知键报错，防止拼写错误静默失效。
def _settings(settings=None):
    supplied = dict(settings or {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown H0 settings: {sorted(unknown)}')
    result = {**DEFAULTS, **supplied}
    if not 0 <= result['om0'] <= 1:
        raise ValueError('om0 must be in [0,1]')
    for key in ('h0_min', 'h0_max', 'h0_step', 'f_max', 'f_step',
                'f_prior_sigma', 'baseline_h0_step', 'baseline_f_step'):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if result['h0_max'] <= result['h0_min']:
        raise ValueError('h0_max must exceed h0_min')
    for key in ('velocity_nodes', 'redshift_error_nodes', 'chunk_h0'):
        if int(result[key]) != result[key] or result[key] < 2:
            raise ValueError(f'{key} must be an integer >=2')
        result[key] = int(result[key])
    if not 0 <= result['max_nonpositive_z_mass'] < 0.5:
        raise ValueError('max_nonpositive_z_mass must be in [0,0.5)')
    return result


# 检查数值域及每源唯一性；红移来源、流场版本等证据由上游 redshift.py 核对。
def validate_inputs(inputs):
    """Validate numeric inputs; provenance/unknown-error choices are upstream."""
    missing = set(REQUIRED) - set(inputs.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    if inputs.empty or inputs.source.isna().any() or inputs.source.duplicated().any():
        raise ValueError('Need nonempty, unique source identifiers')
    for col in REQUIRED[1:]:
        if not np.isfinite(inputs[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{col} contains unknown/nonfinite values; resolve explicitly')
    for col in ('D_mean', 'D_std', 'z_saved', 'z_cmb'):
        if (inputs[col] <= 0).any():
            raise ValueError(f'{col} must be positive')
    for col in ('z_cmb_error', 'sigma_v_kms'):
        if (inputs[col] < 0).any():
            raise ValueError(f'{col} must be nonnegative')


# K(z)=H0 D_L=c(1+z)∫dz/E(z)，单位 km/s；除 H0 后为 Mpc，采用平直 ΛCDM。
def distance_scale(z, om0=0.3):
    """Return H0*D_L=c(1+z)int_0^z dz/E(z), in km/s, for flat LCDM."""
    zz = np.asarray(z, dtype=float)
    if not np.isfinite(zz).all() or np.any(zz < 0) or not 0 <= om0 <= 1:
        raise ValueError('Need finite z>=0 and 0<=om0<=1')
    values = np.array([C_KMS * (1 + float(x)) * quad(
        lambda t: 1 / np.sqrt(om0 * (1 + t)**3 + 1 - om0),
        0, float(x), epsabs=1e-12, epsrel=1e-11)[0] for x in zz.ravel()])
    values = values.reshape(zz.shape)
    return float(values) if zz.ndim == 0 else values


# 外部 Z=z_CMB 与 V 独立高斯，构造 1+z_cos=(1+Z)/(1+V/c) 的比值分布。
# 用解析 CDF/逆 CDF 构造一维 Gauss-Legendre 积分，不线性化 K(z) 或速度分母。
# 负红移概率超阈值时报错；只允许极小负尾被条件化排除，不能泛用于所有低红移源。
def _redshift_distribution(row, pv_enabled, settings, baseline=False, nodes=None):
    """Positive-z quadrature for the ratio of independent Gaussian inputs.

    For Y=1+z_cos=(1+Z)/(1+V/c), denominator-negative probability must
    be below 1e-14. Then F(z)=Phi((a*(1+z)-b)/sqrt(sz^2+s^2*(1+z)^2)).
    Its closed-form inverse reduces velocity and redshift errors to one
    dimension without linearizing K(z) or the Doppler denominator.
    """
    if baseline:
        center = float(row['z_saved'])
        zz, weights = np.array([center]), np.array([1.0])
        return zz, weights, dict(z_center=center, z_q16=center, z_q50=center,
            z_q84=center, z_std=0.0, nonpositive_z_probability=0.0,
            excluded_denominator_probability=0.0, z_error_used=0.0,
            velocity_mean_used=0.0, velocity_sigma_used=0.0,
            redshift_distribution='saved_z_fixed_baseline', quadrature_nodes=1)
    zc, sz = float(row['z_cmb']), float(row['z_cmb_error'])
    vm = float(row['v_flow_kms']) if pv_enabled else 0.0
    sv = float(row['sigma_v_kms']) if pv_enabled else 0.0
    a, b, s = 1 + vm / C_KMS, 1 + zc, sv / C_KMS
    if a <= 0:
        raise ValueError(f"{row['source']}: nonphysical velocity denominator")
    denom_bad = float(ndtr(-a / s)) if s else 0.0
    if denom_bad > 1e-14:
        raise ValueError(f"{row['source']}: Gaussian velocity has non-negligible V<=-c")
    total_sigma = np.hypot(sz, s)
    if total_sigma == 0:
        center = b / a - 1
        if center <= 0:
            raise ValueError(f"{row['source']}: corrected redshift is nonpositive")
        zz, weights, qs, pbad = np.array([center]), np.array([1.]), np.repeat(center, 3), 0.0
    else:
        pbad = float(ndtr((a - b) / total_sigma))
        if pbad > settings['max_nonpositive_z_mass']:
            raise ValueError(f"{row['source']}: P(z_cos<=0)={pbad:.6g} exceeds "
                             f"{settings['max_nonpositive_z_mass']}; requires explicit model handling")
        def from_normal(tt):
            tt = np.asarray(tt)
            denominator = a*a - tt*tt*s*s
            if np.any(denominator <= 0):
                raise ValueError('Ratio-normal quantile outside positive-denominator regime')
            yy = (a*b + tt*np.sqrt(b*b*s*s + sz*sz*denominator)) / denominator
            return yy - 1
        n = int(nodes or settings['velocity_nodes' if pv_enabled else 'redshift_error_nodes'])
        x, w = np.polynomial.legendre.leggauss(n)
        # Integrate in the standard-normal variable, not probability space:
        # the latter has slow endpoint convergence in likelihood tail cases.
        # Nine-sigma finite bounds omit less than 2.3e-19 probability.
        tlo, thi = max((a-b)/total_sigma, -9.0), 9.0
        tt = (thi+tlo)/2 + (thi-tlo)*x/2
        zz = from_normal(tt)
        weights = w*(thi-tlo)/2*norm.pdf(tt)/(1-pbad)
        weights /= weights.sum()
        qs = from_normal(ndtri(pbad+(1-pbad)*np.array([0.15865, 0.5, 0.84135])))
        if np.any(zz <= 0):
            raise ValueError('Positive-redshift quadrature produced a nonpositive node')
    mean = float(np.dot(weights, zz))
    info = dict(z_center=float(qs[1]), z_q16=float(qs[0]), z_q50=float(qs[1]),
        z_q84=float(qs[2]), z_std=float(np.sqrt(np.dot(weights, (zz-mean)**2))),
        nonpositive_z_probability=pbad, excluded_denominator_probability=denom_bad,
        z_error_used=sz, velocity_mean_used=vm, velocity_sigma_used=sv,
        redshift_distribution='positive_truncated_ratio_normal' if pbad else 'ratio_normal',
        normal_quadrature_tail_bound=2.3e-19, quadrature_nodes=len(zz))
    return zz, weights, info


# 每源预计算红移节点、权重和 K(z)，避免整个 H0×f_D 网格重复宇宙学积分。
def _prepare(inputs, pv_enabled, settings, baseline=False, nodes=None):
    prepared = []
    for _, row in inputs.iterrows():
        z, w, info = _redshift_distribution(row, pv_enabled, settings, baseline, nodes)
        prepared.append(dict(row=row.to_dict(), z=z, weights=w, logweights=np.log(w),
            K=distance_scale(z, settings['om0']), info=info))
    return prepared


# 在每个红移节点计算 N(D_mean | K/H0, D_std²+(f_D K/H0)²)，按外部权重求和。
# 方差依赖参数，必须保留 log(2πvariance)；logaddexp 避免小概率下溢。
def _source_ll(h0, f_D, item):
    hh, ff = np.broadcast_arrays(np.asarray(h0, float), np.asarray(f_D, float))
    if np.any(hh <= 0) or np.any(ff < 0):
        raise ValueError('H0 must be positive and f_D nonnegative')
    y, error = float(item['row']['D_mean']), float(item['row']['D_std'])
    result = np.full(hh.shape, -np.inf)
    for k, lw in zip(item['K'], item['logweights']):
        mu = k / hh
        variance = error*error + (ff*mu)**2
        ll = -0.5*((y-mu)**2/variance + np.log(2*np.pi*variance))
        result = np.logaddexp(result, lw+ll)
    return result


# 公开单源似然接口，边缘化外部红移/速度后返回 logL，不写文件。
def source_log_likelihood(h0, f_D, row, pv_enabled=False, settings=None):
    """One AGN likelihood after marginalizing its external redshift/velocity PDF."""
    ss = _settings(settings)
    item = _prepare(pd.DataFrame([dict(row)]), pv_enabled, ss)[0]
    return _source_ll(h0, f_D, item)


# 独立源 logL 求和；此模型未加入跨源相干流场协方差。
def _total_ll(h0, f_D, prepared):
    return sum((_source_ll(h0, f_D, item) for item in prepared), 0.0)


# 含两端点的均匀网格；区间必须为步长整数倍，避免隐性改变积分范围。
def _grid(lo, hi, step):
    count = int(round((hi-lo)/step))
    if count < 2 or not np.isclose(lo+count*step, hi, rtol=0, atol=1e-8):
        raise ValueError('Grid interval must be an integer multiple of its step')
    return np.linspace(lo, hi, count+1)


# 按实际坐标做梯形积分归一化；输出连续密度，不是格点概率。
def _normalized(x, density):
    area = trapezoid(density, x)
    if not np.isfinite(area) or area <= 0:
        raise ValueError('Posterior has no finite positive normalization')
    return density/area


# 累积积分后插值，返回中央 2.275/15.865/50/84.135/97.725% 分位。
def _quantiles(x, pdf):
    cdf = cumulative_trapezoid(pdf, x, initial=0)
    cdf /= cdf[-1]
    return np.interp(PROBS, cdf, x)


# 估计网格高端的积分概率，检查参数上界是否截断显著后验质量。
def _tail(x, pdf, threshold):
    mask = x >= threshold
    return float(trapezoid(pdf[mask], x[mask])) if mask.sum() >= 2 else 0.0


# 计算单情景似然网格并单独优化 MLE；H0 均匀先验，f_D 为半正态先验。
# MLE 不含先验，边缘后验分位数包含先验，二者无需相同。
# 固定 f_D=0 为独立点质量模型，不绘成连续散度密度；返回数组/表，文件由上层保存。
def fit_case(inputs, case, settings=None):
    """Fit one specified branch; returns arrays and tables without writing files."""
    ss = _settings(settings)
    validate_inputs(inputs)
    if case not in CASE_CONFIG:
        raise ValueError(f'Unknown comparison case {case}')
    cfg = CASE_CONFIG[case]
    baseline = case == 'baseline_saved_z'
    if baseline:
        ss = {**ss, 'h0_step': ss['baseline_h0_step'], 'f_step': ss['baseline_f_step']}
    prepared = _prepare(inputs, cfg['pv_enabled'], ss, baseline)
    h = _grid(ss['h0_min'], ss['h0_max'], ss['h0_step'])
    free = cfg['scatter_mode'] == 'fit'
    f = _grid(0., ss['f_max'], ss['f_step']) if free else np.array([0.])
    ll = np.empty((len(h), len(f)))
    for start in range(0, len(h), ss['chunk_h0']):
        stop = min(start+ss['chunk_h0'], len(h))
        ll[start:stop] = _total_ll(h[start:stop, None], f[None, :], prepared)
    maxidx = np.unravel_index(np.argmax(ll), ll.shape)
    # 优化用 log H0 保证正值；后验仍按 H0 积分，先验未改成 log-uniform。
    initial = [np.log(h[maxidx[0]])] + ([f[maxidx[1]]] if free else [])
    bounds = [(np.log(ss['h0_min']), np.log(ss['h0_max']))] + ([(0.,ss['f_max'])] if free else [])
    opt = minimize(lambda theta: -float(_total_ll(np.exp(theta[0]), theta[1] if free else 0., prepared)),
                   initial, method='L-BFGS-B', bounds=bounds,
                   options={'ftol':1e-12, 'gtol':1e-8})
    hm, fm = float(np.exp(opt.x[0])), float(opt.x[1]) if free else 0.
    if free:
        lp = ll - 0.5*(f[None,:]/ss['f_prior_sigma'])**2
        density = np.exp(lp - np.max(lp))
        density /= trapezoid(trapezoid(density, f, axis=1), h)
        ph = _normalized(h, trapezoid(density, f, axis=1))
        pf = _normalized(f, trapezoid(density, h, axis=0))
        fq = _quantiles(f, pf)
    else:
        density, pf = None, None
        ph = _normalized(h, np.exp(ll[:,0]-np.max(ll[:,0])))
        fq = np.zeros(len(PROBS))
    hq = _quantiles(h, ph)
    summary = dict(case=case, **cfg, model='linear_fractional', om0=ss['om0'],
        n_sources=len(inputs), h0_mle=hm, scatter_mle=fm, loglike_mle=-float(opt.fun),
        mle_success=bool(opt.success), mle_message=str(opt.message),
        h0_prior=f"uniform_{ss['h0_min']:g}_{ss['h0_max']:g}",
        scatter_prior=f"halfnormal_{ss['f_prior_sigma']:g}" if free else 'point_mass_at_zero',
        h0_probability_above_90pct_max=_tail(h, ph, 0.9*ss['h0_max']),
        scatter_probability_above_90pct_max=_tail(f, pf, 0.9*ss['f_max']) if free else 0.,
        scatter_prior_probability_above_grid=float(2*norm.sf(ss['f_max']/ss['f_prior_sigma'])) if free else 0.,
        baseline_ignores_redshift_measurement_error=baseline)
    for name, value in zip(QUANTILE_NAMES, hq): summary['h0_'+name] = float(value)
    for name, value in zip(QUANTILE_NAMES, fq): summary['scatter_'+name] = float(value)
    quality = []
    for item in prepared:
        row, w = item['row'], item['weights']
        mu = item['K']/hm
        mean = float(np.dot(w,mu))
        # 全方差分解：节点条件方差的均值，加上节点模型均值变化的方差。
        conditional = float(np.dot(w,float(row['D_std'])**2+(fm*mu)**2))
        mixing = float(np.dot(w,(mu-mean)**2))
        quality.append({**row, **item['info'], 'case':case,
            'D_model_at_mle':mean,
            'D_model_at_z_center_mle':distance_scale(item['info']['z_center'],ss['om0'])/hm,
            'conditional_variance_at_mle':conditional,
            'mixture_variance_at_mle':conditional+mixing,
            'Hubble_residual_sigma':(float(row['D_mean'])-mean)/np.sqrt(conditional+mixing),
            'source_loglike_at_mle':float(_source_ll(hm,fm,item)),
            'residual_interpretation':'mixture mean/variance diagnostic, not a standard chi-square statistic'})
    return dict(summary=summary, settings=ss, H0=h, f_D=f, h0_pdf=ph,
                scatter_pdf=pf, joint_pdf=density, fit_quality=pd.DataFrame(quality))


# 保存情景摘要与诊断为 UTF-8 JSON。
def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')


# 写 H0 边缘密度、可选散度/联合密度和逐源诊断，供绘图直接复用。
def _write_case(folder, result):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'H0':result['H0'],'pdf':result['h0_pdf']}).to_csv(folder/'h0_posterior.csv',index=False)
    if result['scatter_pdf'] is not None:
        pd.DataFrame({'f_D':result['f_D'],'pdf':result['scatter_pdf']}).to_csv(folder/'scatter_posterior.csv',index=False)
        np.savez_compressed(folder/'joint_posterior.npz',H0=result['H0'],f_D=result['f_D'],pdf=result['joint_pdf'])
    result['fit_quality'].to_csv(folder/'fit_quality.csv',index=False)
    _json(folder/'summary.json',{'summary':result['summary'],'settings':result['settings']})


# 独立拟合同一张输入表；baseline_saved_z 是另列的历史红移基线，不混入 A–D。
def run_comparison(inputs, output_dir, *, settings=None, cases=('A','B','C','D'),
                   baseline_saved_z=False, validate=True):
    """Run requested common-input cases; optional original-z baseline is separate."""
    ss = _settings(settings)
    data = inputs.copy(deep=True)
    validate_inputs(data)
    chosen = list(cases)
    if len(chosen) != len(set(chosen)) or any(case not in ('A','B','C','D') for case in chosen):
        raise ValueError('cases must be unique choices from A,B,C,D')
    if baseline_saved_z: chosen.insert(0,'baseline_saved_z')
    if not chosen: raise ValueError('No comparison cases requested')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out/'common_inputs.csv',index=False)
    rows = []
    meta = dict(settings=ss, case_config={case:CASE_CONFIG[case] for case in chosen},
        comparison=rows, input_columns=list(data.columns),
        distance_convention='Frozen saved D_L mean/std; K(z_cos)=c(1+z_cos)int dz/E',
        uncertainty_assumptions='Independent per-source external Gaussian z and velocity inputs; no cross-source velocity covariance',
        velocity_distribution='Normal(v_flow_kms,sigma_v_kms); independent of H0; conditioned on z_cos>0 only below explicit tail threshold',
        validation={'state':'pending' if validate else 'not_requested'})
    _json(out/'run_summary.json',meta)
    for case in chosen:
        print(f'H0 comparison: starting {case}',flush=True)
        result = fit_case(data,case,ss)
        _write_case(out/'cases'/case,result)
        rows.append(result['summary'])
        pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
        _json(out/'run_summary.json',meta)
        print(f"H0 comparison: completed {case}, H0={result['summary']['h0_q50']:.6f}",flush=True)
        del result
    if validate:
        meta['validation'] = validate_saved_comparison(out)
        _json(out/'run_summary.json',meta)
    return meta


# 每隔一点取粗网格并保留末端，用来检查已存密度的分辨率敏感性。
def _coarse_indices(n):
    return np.unique(np.r_[np.arange(0,n,2),n-1])


# 核对情景、输入、归一化、MLE 与粗网格分位数；局部节点加倍检查积分收敛。
# 局部检查仅覆盖中央区间和 MLE 坐标，不保证整个参数域统一精度。
# 边界质量过大只记 advisory，不使验证失败；passed 不代表物理系统误差已受控。
def validate_saved_comparison(output_dir, *, check_quadrature=True):
    """Check saved normalization, grid sensitivity and local node convergence.

    The doubled-node check covers the posterior central interval plus the MLE;
    it is not a second full-grid fit or a proof of uniform quadrature accuracy.
    """
    out = Path(output_dir)
    meta = json.loads((out/'run_summary.json').read_text(encoding='utf-8'))
    inputs = pd.read_csv(out/'common_inputs.csv')
    validate_inputs(inputs)
    expected = meta.get('case_config', {})
    completed = meta.get('comparison', [])
    case_ids = [row.get('case') for row in completed]
    if (not expected or not completed or len(case_ids) != len(set(case_ids))
            or set(case_ids) != set(expected)):
        raise ValueError('Incomplete or duplicate comparison cases: completed set must equal case_config')
    if any(case not in CASE_CONFIG or config != CASE_CONFIG[case]
           for case, config in expected.items()):
        raise ValueError('Saved case_config does not match the defined comparison models')
    reports = []
    for row in meta['comparison']:
        case = row['case']; folder = out/'cases'/case
        saved = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        stored = saved['summary']
        if not row.get('mle_success', False) or not stored.get('mle_success', False):
            raise ValueError(f'{case}: maximum-likelihood optimization did not succeed')
        for key in ('case', 'pv_enabled', 'scatter_mode', 'n_sources', 'model', 'h0_prior', 'scatter_prior'):
            if stored.get(key) != row.get(key):
                raise ValueError(f'{case}: case summary differs from run summary: {key}')
        if row['n_sources'] != len(inputs):
            raise ValueError(f'{case}: wrong number of source constraints')
        for key in ('pv_enabled', 'scatter_mode'):
            if row[key] != expected[case][key]:
                raise ValueError(f'{case}: case summary differs from its model definition')
        for key in ('h0_mle', 'scatter_mle', 'loglike_mle', 'om0',
                    'h0_q16', 'h0_q50', 'h0_q84', 'scatter_q16', 'scatter_q50', 'scatter_q84'):
            if (key not in stored or key not in row or not np.isfinite(row[key])
                    or not np.isclose(stored[key], row[key], rtol=1e-12, atol=1e-12)):
                raise ValueError(f'{case}: case summary differs from run summary: {key}')
        ss = _settings(saved['settings'])
        d = pd.read_csv(folder/'h0_posterior.csv'); h=d.H0.to_numpy(); ph=d.pdf.to_numpy()
        if not np.isclose(trapezoid(ph,h),1.,atol=1e-8): raise ValueError(f'{case}: H0 PDF not normalized')
        ix = _coarse_indices(len(h))
        free = row['scatter_mode']=='fit'
        if free:
            with np.load(folder/'joint_posterior.npz') as archive:
                hh=archive['H0']; f=archive['f_D']; joint=archive['pdf']
            if not np.array_equal(h,hh): raise ValueError(f'{case}: inconsistent H0 grids')
            if not np.isclose(trapezoid(trapezoid(joint,f,axis=1),h),1.,atol=1e-8):
                raise ValueError(f'{case}: joint PDF not normalized')
            pf=trapezoid(joint,h,axis=0); jx=_coarse_indices(len(f)); coarse=joint[np.ix_(ix,jx)]
            ch=_quantiles(h[ix],trapezoid(coarse,f[jx],axis=1))
            cf=_quantiles(f[jx],trapezoid(coarse,h[ix],axis=0))
            df=float(np.max(np.abs(cf-_quantiles(f,pf))))
        else:
            ch=_quantiles(h[ix],ph[ix]); df=0.
        dh=float(np.max(np.abs(ch-_quantiles(h,ph))))
        if dh>ss['grid_h0_tolerance'] or df>ss['grid_scatter_tolerance']:
            raise ValueError(f'{case}: coarse-grid sensitivity exceeds tolerance: H0={dh}, f={df}')
        quality=pd.read_csv(folder/'fit_quality.csv')
        for col in REQUIRED:
            if col=='source':
                equal=quality[col].tolist()==inputs[col].tolist()
            else:
                equal=np.array_equal(quality[col].to_numpy(),inputs[col].to_numpy())
            if not equal: raise ValueError(f'{case}: common input column changed: {col}')
        maxdelta=0.
        if check_quadrature and case!='baseline_saved_z':
            pv=row['pv_enabled']; n=ss['velocity_nodes' if pv else 'redshift_error_nodes']
            base=_prepare(inputs,pv,ss,nodes=n); fine=_prepare(inputs,pv,ss,nodes=2*n)
            hs=[row['h0_q16'],row['h0_q50'],row['h0_q84'],row['h0_mle']]
            fs=[row['scatter_q16'],row['scatter_q50'],row['scatter_q84'],row['scatter_mle']] if free else [0.]
            maxdelta=max(abs(float(_total_ll(hv,fv,base)-_total_ll(hv,fv,fine))) for hv in hs for fv in fs)
            if maxdelta>ss['quadrature_loglike_tolerance']:
                raise ValueError(f'{case}: doubled-node local log-likelihood change {maxdelta:.6g} exceeds tolerance')
        tail_diagnostics = {
            'h0_probability_above_90pct_max': _tail(h, ph, 0.9*ss['h0_max']),
            'scatter_probability_above_90pct_max': _tail(f, pf, 0.9*ss['f_max']) if free else 0.,
            'scatter_prior_probability_above_grid': float(2*norm.sf(ss['f_max']/ss['f_prior_sigma'])) if free else 0.,
        }
        notes = []
        if max(tail_diagnostics.values()) > 0.01:
            notes.append('More than one percent probability reaches a grid boundary band or lies beyond the scatter prior grid; review a larger grid. This is advisory, not a hard validation failure.')
        report=dict(case=case,state='passed',passed=True,h0_coarse_grid_quantile_maxdiff=dh,
            scatter_coarse_grid_quantile_maxdiff=df,local_doubled_node_loglike_maxdiff=maxdelta,
            quadrature_check_scope='central 68.27 percent H0/scatter Cartesian product plus MLE coordinates',
            max_nonpositive_z_probability=float(quality.nonpositive_z_probability.max()),
            common_inputs_unchanged=True,tail_diagnostics=tail_diagnostics,advisory_notes=notes,
            not_a_chi_square_goodness_of_fit_test=True)
        reports.append(report)
        _json(folder/'numerical_validation.json',report)
    report=dict(state='passed',passed=True,cases=reports,number_of_sources=len(inputs),
        fixed_zero_cases_are_separate_models=True)
    _json(out/'numerical_validation.json',report)
    return report

