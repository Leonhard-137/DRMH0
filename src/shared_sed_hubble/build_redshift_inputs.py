# 外部证据准备脚本：NED 原始 VOTable + 固定版本 2M++ 流场，和离线 H0 拟合分开。
# Astropy SkyCoord 负责 ICRS/超星系坐标；urllib 获取公开数据；XML 解析器核对唯一对象。
# 未加 --refresh 但缓存缺失时仍可能联网；本次导读不执行此入口。
"""Rebuild host redshift / pinned 2M++ inputs. Use --refresh for public retrieval.

Only NED's current page-facing summary endpoint is queried. Raw VOTables are
retained. The complete LFS grid is retrieved and checked against its pinned SHA256;
only cells queried by this sample and one regression fixture are retained.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, hashlib, io, json, sys, subprocess
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import numpy as np
import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = Path(__file__).resolve().parents[2] / 'data' / 'peculiar_velocity'
C = 299792.458
COMMIT = '2b016e9ae281aff1b35f3691d293f2d6a9020e13'
RAW = f'https://raw.githubusercontent.com/KSaid-1/pvhub/{COMMIT}/'
MEDIA = f'https://media.githubusercontent.com/media/KSaid-1/pvhub/{COMMIT}/'
ENDPOINT = 'https://ned.ipac.caltech.edu/sites/all/modules/custom/services/services/summary.php'
NAMES = {'F9_a':'Fairall 9', 'Mrk142':'Mrk 142', 'Mrk279_b':'Mrk 279',
         'Mrk335':'Mrk 335', 'Mrk509':'Mrk 509', 'Mrk817_a':'Mrk 817',
         'NGC2617_a':'NGC 2617', 'NGC4151_a':'NGC 4151',
         'NGC5548':'NGC 5548', 'NGC6814':'NGC 6814'}
NS = {'v':'http://www.ivoa.net/xml/VOTable/v1.3'}
NETWORK_PYTHON = None

# 保留 HTTPS 证书校验；可选解释器仅承载网络传输，不改系统代理/TLS 设置。
def fetch(url, data=None):
    if NETWORK_PYTHON:
        # Explicit alternate interpreter for HTTPS transport only; certificates
        # remain verified and no machine-wide proxy/TLS settings are changed.
        code = "import sys,urllib.request; u=sys.argv[1]; d=bytes.fromhex(sys.argv[2]) if sys.argv[2] else None; r=urllib.request.urlopen(urllib.request.Request(u,data=d,headers={'User-Agent':'AGN-host-redshift-audit/1.0'}),timeout=60); sys.stdout.buffer.write(r.read())"
        return subprocess.check_output([NETWORK_PYTHON,'-c',code,url,data.hex() if data is not None else ''])
    req = urllib.request.Request(url, data=data, headers={'User-Agent':'AGN-host-redshift-audit/1.0'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

# 保存请求、流场摘要和证据 JSON，便于离线追溯。
def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

# NED 必须仅返回一个 OK 表和唯一对象，拒绝模糊搜索结果。
def parse_xml(raw):
    root = ET.fromstring(raw)
    tables = root.findall('.//v:TABLE',NS)
    if len(tables) != 1:
        raise ValueError('NED response must have exactly one table')
    table = tables[0]
    statuses = [p.get('value') for p in table.findall('.//v:PARAM',NS) if p.get('name')=='QUERY_STATUS']
    if statuses != ['OK']:
        raise ValueError(f'NED QUERY_STATUS {statuses}')
    fields = table.findall('v:FIELD',NS)
    records = table.findall('.//v:TR',NS)
    if len(records) != 1:
        raise ValueError('Expected one unambiguous NED object')
    row = dict(zip([f.get('ID',f.get('name')) for f in fields], [td.text for td in records[0]]))
    return row

# 读取/获取宿主红移与坐标原始响应，记录请求和 UTC 时间。
def ned_snapshot(source, target, refresh):
    out={}
    for kind,key in [('RedshiftSummaryOfObject','redshift'),('PositionSummaryOfObject','position')]:
        path=ROOT/'raw'/f'{source}_{key}.xml'
        params='?'+urllib.parse.urlencode({'TARGET':target,'hconst':67.8,'omegam':.308,'omegav':.692,'corr_z':1})
        payload={'params_url':params,'service_url':'http://opslas3.ned.ipac.caltech.edu/services/ui/'+kind}
        if refresh or not path.exists():
            raw=json.loads(fetch(ENDPOINT,urllib.parse.urlencode(payload).encode()))['result']
            parse_xml(raw)
            path.write_text(raw,encoding='utf-8')
            dump(path.with_suffix('.request.json'),{'public_endpoint':ENDPOINT,'method':'POST','form':payload,'retrieved_utc':datetime.now(timezone.utc).isoformat()})
        out[key]=parse_xml(path.read_text(encoding='utf-8'))
    print('NED snapshot',source,'verified',flush=True)
    return source,out

# 用 cz（km/s）作半径，转到超星系坐标后选最近的 129³ 网格单元。
# 这里的 distance 为速度坐标，不是 Mpc；覆盖判断同时考虑半径和三轴界限。
def grid_position(ra,dec,zcmb):
    sky=SkyCoord(np.asarray(ra)*u.degree,np.asarray(dec)*u.degree,distance=C*np.asarray(zcmb)*u.km/u.s,frame='icrs')
    xyz=sky.transform_to('supergalactic').cartesian.xyz.value
    bins=np.round((xyz+20000.)/312.5).astype(int)
    index=bins[0]*129*129+bins[1]*129+bins[2]
    inside=(C*np.asarray(zcmb)<19942.) & (np.abs(xyz)<20000.).all(axis=0)
    return xyz,bins,index,inside

# 获取固定 commit 支持文件；Git LFS 载荷必须匹配长度和 SHA256。
def checked_small_file(remote,local,refresh,lfs=False):
    pointer=fetch(RAW+remote).decode() if refresh or not local.exists() else None
    if pointer and pointer.startswith('version https://git-lfs.github.com/spec/v1'):
        sha=pointer.split('oid sha256:')[1].splitlines()[0]
        size=int(pointer.split('size ')[1].splitlines()[0])
        content=fetch(MEDIA+remote)
        if len(content)!=size or hashlib.sha256(content).hexdigest()!=sha:
            raise ValueError('Pinned LFS payload integrity mismatch: '+remote)
    elif pointer is not None:
        content=pointer.encode()
    if pointer is not None:
        local.write_bytes(content)
    return {'path':str(local.relative_to(ROOT)),'url':MEDIA+remote if lfs else RAW+remote,'sha256':hashlib.sha256(local.read_bytes()).hexdigest(),'size_bytes':local.stat().st_size}

# 完整下载并核对固定流场，磁盘只保留指定单元；缺缓存时约下载 118.6 MB。
def grid_subset(indices,refresh):
    path=ROOT/'model'/'2MPP_redshift_subset.csv'
    manifest_path=ROOT/'model'/'grid_manifest.json'
    if not refresh and path.exists() and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['subset_sha256']:
            raise ValueError('Local grid subset does not match retained manifest')
        rows={int(r['grid_index']):r for r in csv.DictReader(path.open())}
        if set(indices)<=rows.keys():return rows,manifest
    pointer=fetch(RAW+'data/2MPP_redshift.txt').decode()
    sha=pointer.split('oid sha256:')[1].splitlines()[0]
    size=int(pointer.split('size ')[1].splitlines()[0])
    h=hashlib.sha256(); count=0; rows={}; lines=0
    # Hash all original bytes, retaining only specified rows on disk.
    print('Retrieving pinned 118.6 MB flow grid for SHA256-verified subset', flush=True)
    with io.BytesIO(fetch(MEDIA+'data/2MPP_redshift.txt')) as f:
        header=f.readline();h.update(header);count+=len(header)
        fields=next(csv.reader([header.decode().strip()]))
        for i,line in enumerate(f):
            h.update(line);count+=len(line);lines+=1
            if i in indices:
                vals=next(csv.reader([line.decode().strip()]))
                rows[i]=dict(zip(fields,vals));rows[i]['grid_index']=i
    if count!=size or h.hexdigest()!=sha or lines!=129**3:
        raise ValueError(f'Grid integrity failure {count} {h.hexdigest()} {lines}')
    if set(indices)-rows.keys():raise ValueError('Requested grid indices missing')
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['grid_index']+fields);writer.writeheader();writer.writerows(rows[i] for i in sorted(rows))
    manifest={'commit':COMMIT,'url':MEDIA+'data/2MPP_redshift.txt','lfs_sha256':sha,'size_bytes':size,'data_rows':lines,'selected_indices':sorted(indices),'subset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    dump(manifest_path,manifest)
    return rows,manifest

# 对原始 NED、保留流场和乘性红移往返作独立核对。
# 会加载保留的上游代码并写验证 JSON；本次仅静态阅读，不执行。
def validate_snapshot():
    """Check retained input evidence independently of network acquisition."""
    from .redshift import load_redshift_inputs, correct_redshift
    frame=load_redshift_inputs(ROOT/'source_inputs.csv', list(NAMES))
    vmax=0.;emax=0.
    for source in NAMES:
        r=parse_xml((ROOT/'raw'/f'{source}_redshift.xml').read_text())
        p=parse_xml((ROOT/'raw'/f'{source}_position.xml').read_text())
        row=frame.set_index('source').loc[source]
        assert np.isclose(row.z_hel,float(r['z']),atol=1e-15,rtol=0)
        assert np.isclose(row.z_cmb*C,float(r['v_Sun_3K']),atol=1e-8,rtol=0)
        assert np.isclose(row.z_cmb_error*C,float(r['unc_v_Sun_3K']),atol=1e-8,rtol=0)
        assert np.isclose(row.ra_deg,float(p['equ_j2000_lon']),atol=1e-10,rtol=0)
        assert np.isclose(row.dec_deg,float(p['equ_j2000_lat']),atol=1e-10,rtol=0)
        lon,lat=np.radians([float(p['gal_lon']),float(p['gal_lat'])])
        la,ba=np.radians([264.14,48.26])
        delta=371*(np.sin(lat)*np.sin(ba)+np.cos(lat)*np.cos(ba)*np.cos(lon-la))
        observed=float(r['v_Sun_3K'])-float(r['v_Sun'])
        vmax=max(vmax,abs(delta-observed))
        emax=max(emax,abs(np.hypot(float(r['unc_v_Sun']),.07*observed)-float(r['unc_v_Sun_3K'])))
    assert vmax<1e-9 and emax<1e-9
    assert np.array_equal(correct_redshift(frame.z_cmb,np.zeros(len(frame))),frame.z_cmb.to_numpy())
    recovered=(1+correct_redshift(frame.z_cmb,frame.v_flow_kms))*(1+frame.v_flow_kms/C)-1
    roundtrip=float(np.max(abs(recovered-frame.z_cmb)))
    assert roundtrip<3e-16
    # Run the retained upstream implementation on the exact retained cells.
    import importlib.util
    import pandas as pd
    spec=importlib.util.spec_from_file_location('pinned_pvhub',ROOT/'model'/'pvhub_upstream.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    recon=upstream.TwoMPP_redshift.__new__(upstream.TwoMPP_redshift)
    recon.dmin=-20000.;recon.dmax=20000.;recon.nbins=129;recon.bsz=312.5
    subset=pd.read_csv(ROOT/'model'/'2MPP_redshift_subset.csv')
    grid=np.full(129**3,np.nan)
    grid[subset.grid_index.to_numpy(int)]=subset.vproj_2MPP.to_numpy(float)
    recon.vmodel=pd.DataFrame({'vproj_2MPP':grid})
    recon.vextmodel=pd.read_csv(ROOT/'model'/'2MPP_redshift_out.txt',sep=r'\s+')
    got=recon.calculate_pv(frame.ra_deg.to_numpy(),frame.dec_deg.to_numpy(),frame.z_cmb.to_numpy())
    assert np.array_equal(got,frame.v_flow_kms.to_numpy())
    report={'status':'passed','sources':list(NAMES),'raw_NED_rows_match_snapshot':True,
            'ned_cmb_projection_max_residual_kms':vmax,'ned_cmb_error_formula_max_residual_kms':emax,
            'multiplicative_inverse_max_residual_z':roundtrip,'pinned_upstream_flow_all_ten_match':True,
            'all_sources_in_grid':bool(frame.flow_branch.eq('grid_nearest').all()),
            'evidence_level':'numerical-evidence'}
    dump(ROOT/'validation.json',report)
    print(json.dumps(report,indent=2))

# 刷新或补齐十源快照，采用 sigma_v=250 km/s 的剩余速度分析假设。
# 流场外只用发布的外部偶极表；未定义/越界即报错，不填零。
def main():
    global NETWORK_PYTHON, ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');parser.add_argument('--output-dir',type=Path,default=ROOT);parser.add_argument('--validate-only',action='store_true');parser.add_argument('--network-python',help='Optional interpreter for verified HTTPS transport only');args=parser.parse_args()
    NETWORK_PYTHON=args.network_python
    ROOT=args.output_dir.resolve()
    if args.validate_only:
        validate_snapshot()
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    for name in ['raw','model']:(ROOT/name).mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda kv:ned_snapshot(*kv,args.refresh),NAMES.items()))
    records=[]
    for source,info in results:
        r=info['redshift'];p=info['position']
        required=['z','unc_z','v_Sun','unc_v_Sun','v_Sun_3K','unc_v_Sun_3K','z_refcode']
        if any(r.get(k) is None for k in required):raise ValueError(f'Missing measurement/error for {source}')
        if r['z_frame']!='3K CMB':raise ValueError('NED returned incorrect frame')
        if not np.isclose(float(r['z'])*C,float(r['v_Sun']),rtol=0,atol=1e-7):raise ValueError('NED cz convention mismatch')
        if not np.isclose(float(r['z_corr'])*C,float(r['v_Sun_3K']),rtol=0,atol=1e-7):raise ValueError('NED CMB convention mismatch')
        rec=dict(source=source,ned_name=NAMES[source],ra_deg=float(p['equ_j2000_lon']),dec_deg=float(p['equ_j2000_lat']),z_hel=float(r['z']),z_hel_error=float(r['unc_z']),z_cmb=float(r['v_Sun_3K'])/C,z_cmb_error=float(r['unc_v_Sun_3K'])/C,v_hel_kms=float(r['v_Sun']),v_hel_error_kms=float(r['unc_v_Sun']),v_cmb_kms=float(r['v_Sun_3K']),v_cmb_error_kms=float(r['unc_v_Sun_3K']),sigma_v_kms=250.,input_kind='host',redshift_reference=r['z_refcode'],coordinate_reference=p['refcode'],flow_model='pvhub.TwoMPP_redshift',flow_version=COMMIT,measurement_error_status='NED_reported',cmb_error_status='NED_reported_includes_frame_correction',redshift_technique=r['z_technique'],ned_velocity_note=r.get('v_note') or '',query_status='pending_flow')
        records.append(rec)
    xyz,bins,idx,inside=grid_position([r['ra_deg'] for r in records],[r['dec_deg'] for r in records],[r['z_cmb'] for r in records])
    # Fixed historical fixture verifies bin order independently of live NED updates.
    fixture_xyz,fixture_bins,fixture_idx,_=grid_position(182.6357333,39.405851,0.004149239774)
    if int(fixture_idx)!=1090501:raise ValueError('Supergalactic conversion regression failed')
    needed=set(int(i) for i,flag in zip(idx,inside) if flag);needed.add(int(fixture_idx))
    subset,grid_meta=grid_subset(needed,args.refresh)
    small=[]
    for remote,name,lfs in [('pvhub.py','pvhub_upstream.py',False),('README.md','pvhub_README.md',False),('data/2MPP_redshift_out.txt','2MPP_redshift_out.txt',True)]:
        small.append(checked_small_file(remote,ROOT/'model'/name,args.refresh,lfs))
    ext=np.genfromtxt(ROOT/'model'/'2MPP_redshift_out.txt',names=True)
    for n,r in enumerate(records):
        raw_v=float(subset[int(idx[n])]['vproj_2MPP']) if inside[n] else np.nan
        branch='grid_nearest'
        if not inside[n] or (not np.isfinite(raw_v) and r['z_cmb']>.01):
            k=int(np.searchsorted(ext['z'],r['z_cmb']))
            if k>=len(ext):raise ValueError('Outside tabulated extrapolation range: '+r['source'])
            raw_v=float(np.dot(xyz[:,n],[ext['Vsgx'][k],ext['Vsgy'][k],ext['Vsgz'][k]])/np.linalg.norm(xyz[:,n]))
            branch='published_external_dipole'
        if not np.isfinite(raw_v):raise ValueError('Undefined flow cell: '+r['source'])
        r.update(v_flow_kms=float(np.round(raw_v)),v_flow_unrounded_kms=raw_v,flow_branch=branch,flow_grid_index=int(idx[n]),flow_xbin=int(bins[0,n]),flow_ybin=int(bins[1,n]),flow_zbin=int(bins[2,n]),z_cos=(1+r['z_cmb'])/(1+float(np.round(raw_v))/C)-1,query_status='verified')
    fixture_v=float(subset[int(fixture_idx)]['vproj_2MPP'])
    if not np.isclose(fixture_v,-161.102405,atol=1e-6):raise ValueError('Frozen flow-cell regression failed')
    fields=list(records[0])
    with (ROOT/'source_inputs.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    prov={'created_utc':datetime.now(timezone.utc).isoformat(),'source_policy':'NED current fiducial host heliocentric redshift and NED 3K CMB values; no adopted groups; no upstream z_saved substituted','NED_page_url_template':'https://ned.ipac.caltech.edu/byname?objname={name}&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1','NED_public_endpoint':ENDPOINT,'NED_cmb_convention':'cz optical convention; z_cmb=v_Sun_3K/c. NED traditional additive frame correction; use returned NED uncertainty, not an additional duplicate dipole error.','NED_help':'https://ned.ipac.caltech.edu/Documents/Guides/Calculators','NED_cmb_coefficients':{'apex_speed_kms':371.,'apex_galactic_l_deg':264.14,'apex_galactic_b_deg':48.26,'delta_v':'371*[sin(b)*sin(b_apex)+cos(b)*cos(b_apex)*cos(l-l_apex)]','error_formula':'sqrt(sigma_vhel^2+(0.07*delta_v)^2)','verification':'All 10 returned NED velocities and errors reproduced from raw Galactic coordinates to <1e-9 km/s; no second correction or error term applied'},'flow':{'model':'TwoMPP_redshift','commit':COMMIT,'grid':grid_meta,'supporting_files':small,'grid_method':'Original pvhub nearest supergalactic cell, numpy round to nearest integer km/s; cz input in km/s; no real-space distance input','boundary':'cz<19942 and each Cartesian coordinate strictly inside (-20000,20000) km/s; undefined high-z cells use published external dipole table; unsupported values fail','reference_urls':['https://arxiv.org/abs/1504.04627','https://arxiv.org/abs/2112.01471']},'residual_velocity':{'sigma_kms':250.,'status':'analysis_assumption','distribution':'Gaussian line-of-sight residual about predicted flow','scope':'host velocity branch; common fixed residual assumption, not a measured per-object flow error','not_included':'spatial covariance and flow-model parameter uncertainty are not separately marginalized'},'redshift_equation':'1+z_cos=(1+z_cmb)/(1+v_flow/c), c=299792.458 km/s; low-velocity multiplicative convention consistent with Carrick eq15','NGC4151_group_candidates_not_adopted':[{'catalog':'Crook2007 HDC','group':706,'N':85,'mean_heliocentric_cz_kms':795,'dispersion_kms':253.,'source':'https://cdsarc.cds.unistra.fr/ftp/J/ApJ/655/790/table7.dat'},{'catalog':'Crook2007 LDC','group':867,'N':138,'mean_heliocentric_cz_kms':1005,'dispersion_kms':438.9,'source':'https://cdsarc.cds.unistra.fr/ftp/J/ApJ/655/790/table6.dat'}],'software':{'python':sys.version,'numpy':np.__version__,'astropy':astropy.__version__},'verification':{'ngc4151_fixture_grid_index':int(fixture_idx),'ngc4151_fixture_bins':fixture_bins.tolist(),'ngc4151_fixture_v_unrounded_kms':fixture_v,'full_grid_sha256_verified':True,'all_rows_have_reported_measurement_errors':True,'all_sources':list(NAMES),'evidence_level':'numerical-evidence'}}
    dump(ROOT/'provenance.json',prov)
    for r in records:print(r['source'],r['z_hel'],r['z_cmb'],r['v_cmb_error_kms'],r['v_flow_kms'],r['flow_branch'],r['redshift_reference'])
    print('WROTE',ROOT/'source_inputs.csv',flush=True)
if __name__=='__main__':main()
