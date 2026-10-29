# 只消费已验证的外部 CSV 快照；拟合不联网，也不以旧 z 填补缺失宿主红移。
# 正 v_flow 为额外退行速度；sigma_v 在似然中边缘化，不加到中心校正值。
"""Verified host redshifts and a fixed 2M++ redshift-space correction.

NED tabulates optical-convention cz in its CMB column; inputs are snapshots,
not a network lookup during inference. The residual velocity dispersion is an
analysis assumption and must be marginalized by the H0 likelihood, not added to
this deterministic central correction.
"""
from pathlib import Path
import numpy as np
import pandas as pd

C_KMS = 299792.458
FLOW_VERSION = '2b016e9ae281aff1b35f3691d293f2d6a9020e13'
FLOW_MODEL = 'pvhub.TwoMPP_redshift'
REQUIRED_COLUMNS = (
    'source', 'ra_deg', 'dec_deg', 'z_hel', 'z_cmb', 'z_cmb_error',
    'v_flow_kms', 'sigma_v_kms', 'input_kind', 'redshift_reference',
    'coordinate_reference', 'flow_model', 'flow_version', 'query_status',
    'measurement_error_status', 'flow_branch',
)


# 计算 (1+z_CMB)/(1+v_flow/c)-1；采用低速特殊红移约定，不是完整相对论多普勒逆变换。
def correct_redshift(z_cmb, v_flow_kms):
    """Remove positive receding flow using Carrick+2015 eq.15 convention.

    Return (1 + z_cmb) / (1 + v_flow/c) - 1. This retains the
    multiplicative redshift factors while treating v_flow/c as the low-velocity
    peculiar redshift, matching the reconstruction convention. It is not a
    full relativistic Doppler inversion. Negative corrected redshifts are not
    silently clipped; a downstream physical likelihood must handle support.
    """
    z, velocity = np.broadcast_arrays(np.asarray(z_cmb, dtype=float),
                                      np.asarray(v_flow_kms, dtype=float))
    if not np.isfinite(z).all() or not np.isfinite(velocity).all():
        raise ValueError('Redshift and flow must be finite')
    if np.any(z <= -1) or np.any(np.abs(velocity) >= C_KMS):
        raise ValueError('Invalid redshift or peculiar-velocity domain')
    result = (z - velocity / C_KMS) / (1 + velocity / C_KMS)
    return float(result) if result.ndim == 0 else result


# 核对来源状态、固定流场版本、单位、坐标，并按请求顺序返回。
# 已有 z_cos 也需满足校正式；未知误差、未覆盖流场和群平均不能静默替换。
def load_redshift_inputs(path, sources=None):
    """Read verified CSV, failing closed on incomplete or unsupported inputs.

    Returns a DataFrame with its source column intact. If ``sources`` is given,
    select exactly those rows and preserve their requested order. No saved
    science-pipeline redshift, missing uncertainty, out-of-range zero flow, or
    group mean is substituted. Acquisition evidence is in the adjacent
    provenance.json and raw VOTables; rebuilding is an explicit separate step.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'PV input snapshot is required: {path}')
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f'PV input columns missing: {missing}')
    if frame.empty or frame.source.isna().any() or frame.source.duplicated().any():
        raise ValueError('PV inputs must contain unique, named sources')
    if sources is not None:
        sources = list(sources)
        if len(sources) != len(set(sources)):
            raise ValueError('Requested PV sources must be unique')
        absent = sorted(set(sources) - set(frame.source))
        if absent:
            raise ValueError(f'No verified PV inputs for: {absent}')
        frame = frame.set_index('source').loc[sources].reset_index()
    checks = {
        'query_status': {'verified'},
        'input_kind': {'host'},
        'flow_model': {FLOW_MODEL},
        'flow_version': {FLOW_VERSION},
        'measurement_error_status': {'NED_reported'},
        'flow_branch': {'grid_nearest', 'published_external_dipole'},
    }
    for column, allowed in checks.items():
        bad = frame.loc[~frame[column].isin(allowed), 'source'].tolist()
        if bad:
            raise ValueError(f'Unverified/unsupported {column} for {bad}')
    for column in ['redshift_reference', 'coordinate_reference']:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing provenance: {column}')
    numeric = ['ra_deg', 'dec_deg', 'z_hel', 'z_cmb', 'z_cmb_error',
               'v_flow_kms', 'sigma_v_kms']
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError('PV numerical inputs must all be finite')
    if not frame.ra_deg.between(0, 360, inclusive='left').all():
        raise ValueError('RA must be in [0, 360) degrees')
    if not frame.dec_deg.between(-90, 90).all():
        raise ValueError('Declination must be in [-90, 90] degrees')
    if (frame[['z_hel','z_cmb','z_cmb_error','sigma_v_kms']] <= 0).any().any():
        raise ValueError('Redshifts and reported uncertainties must be positive')
    corrected = correct_redshift(frame.z_cmb, frame.v_flow_kms)
    if np.any(corrected <= 0):
        raise ValueError('Central flow correction gives nonpositive cosmological redshift')
    if 'z_cos' in frame and not np.allclose(frame.z_cos, corrected, rtol=0, atol=1e-12):
        raise ValueError('Cached z_cos is inconsistent with the pinned correction')
    frame['z_cos'] = corrected
    return frame
