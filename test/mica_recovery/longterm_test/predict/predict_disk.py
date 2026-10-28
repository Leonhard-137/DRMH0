"""Numerical design probe for the full-459-day Mrk817 Gaussian/Gamma targets.
Does not generate light curves, launch MICA, or change native run artifacts.
Signal amplitudes use fractional-flux units. Gamma widths reported as standard
deviations (sqrt(2)*native scale). Empirical continuum prior is a Hessian
approximation, not a sampled MICA posterior. Local modes are not posterior medians.
"""

import sys,time,json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
EXPERIMENT_ROOT=Path(__file__).resolve().parent.parent
ROOT=EXPERIMENT_ROOT.parents[2]
sys.path.insert(0,str(EXPERIMENT_ROOT))
from prepare import gaussian_ou_covariance as Gcov
base=ROOT/Path('Mrk817/runs/mica/gaussian/free/2comp/run_UVW2_to_U_2comp_gaussian')
obs=[np.loadtxt(base/'input'/v) for v in ('cont.txt','line.txt')]
t0=min(z[0,0] for z in obs)
tw,tu=[z[:,0]-t0 for z in obs]
ew,eu=[z[:,2]/z[:,1].mean() for z in obs]
dw=tw[:,None]-tw[None,:];du=tu[:,None]-tu[None,:];h=tu[None,:]-tw[:,None]
nw,nu=len(tw),len(tu);n=nw+nu
L=np.zeros((n,2));L[:nw,0]=1;L[nw:,1]=1
noise=np.diag(np.r_[ew**2,eu**2])
# sampled covariance via Fourier inversion, OU driver covariance exact
dt=.1; N=2**18
freq=2*np.pi*np.fft.rfftfreq(N,dt)
def fft_cov(psd,lag):
    cf=np.fft.irfft(psd,n=N)/dt
    v=np.mod(lag/dt,N); k=np.floor(v).astype(int); f=v-k
    return cf[k]*(1-f)+cf[(k+1)%N]*f
def modelcov(q,kind):
    sig,tau=np.exp(q[:2]);a=np.exp(q[[2,5]]);m=q[[3,6]];w=np.exp(q[[4,7]])
    P=2*sig**2*tau/(1+(freq*tau)**2)
    if kind=='gaussian':
        k=lambda lag,s: sig**2*Gcov(lag,s,tau)
        xy=sum(a[j]*k(h-m[j],w[j]) for j in range(2))
        yy=sum(a[j]*a[i]*k(du-m[j]+m[i],np.hypot(w[i],w[j])) for i in range(2) for j in range(2))
    else:
        theta=w/np.sqrt(2)
        H=sum(a[j]*np.exp(-1j*freq*(m[j]-2*theta[j]))/(1+1j*freq*theta[j])**2 for j in range(2))
        xy=fft_cov(P*H,h)
        yy=fft_cov(P*np.abs(H)**2,du)
    xx=sig**2*np.exp(-np.abs(dw)/tau)
    return np.block([[xx,xy],[xy.T,yy]])+noise
def loss(C,Ctrue,actual=None):
    cf=cho_factor(C,lower=True,check_finite=False)
    logdet=2*np.log(np.diag(cf[0])).sum()
    iL=cho_solve(cf,L,check_finite=False); Z=L.T@iL
    ld=np.linalg.slogdet(Z)[1]
    if actual is None:
        v=cho_solve(cf,Ctrue,check_finite=False)
        trace=np.trace(v)-np.trace(np.linalg.solve(Z,iL.T@Ctrue@iL))
    else:
        iy=cho_solve(cf,actual,check_finite=False)
        v=L.T@iy
        trace=actual@iy-v@np.linalg.solve(Z,v)
    return .5*(logdet+ld+trace)
def qpars(sig,tau,a0,m0,s0,a1,m1,s1):
    return np.array([np.log(sig),np.log(tau),np.log(a0),m0,np.log(s0),np.log(a1),m1,np.log(s1)])
starts={
'gaussian':[qpars(.26,150,.787,.204,.205,.386,11.26,29.08),qpars(.26,150,.8,.2,.2,.02,30,80),qpars(.26,150,.8,.2,.2,2,5,145)],
'gamma':[qpars(.26,150,.833,.167,.334,7.137,4.30,213.7),qpars(.26,150,.8,.2,.3,.35,20,25),qpars(.26,150,.8,.2,.3,.01,30,80)]}
def getbounds(kind):
    slo=.14195358*(np.sqrt(2) if kind=='gamma' else 1)
    shi=152.84*(np.sqrt(2) if kind=='gamma' else 1)
    return [(np.log(.05),np.log(1.5)),(np.log(5),np.log(3000)),(np.log(.05),np.log(3)),(0,2),(np.log(slo),np.log(2)),(np.log(1e-5),np.log(10)),(0,100),(np.log(slo),np.log(shi))]
def reportq(q):
    return dict(sig=np.exp(q[0]),tau=np.exp(q[1]),a0=np.exp(q[2]),mu0=q[3],s0=np.exp(q[4]),a1=np.exp(q[5]),mu1=q[6],s1=np.exp(q[7]))
def project(Ctrue,kinds=('gaussian','gamma'),maxiter=80):
    results=[]
    for kind in kinds:
        for j,x in enumerate(starts[kind]):
            begin=time.time()
            opt=minimize(lambda q:loss(modelcov(q,kind),Ctrue),x,method='L-BFGS-B',bounds=getbounds(kind),options={'maxiter':maxiter,'ftol':1e-8,'maxls':15})
            out={'kind':kind,'start':j,'loss':opt.fun,'success':bool(opt.success),'nfev':opt.nfev,'seconds':time.time()-begin,**reportq(opt.x)}
            print(json.dumps(out),flush=True);results.append(out)
    return results

import ctypes as ct
lib=ct.CDLL('/home/leonhard/software/MICA2-parallel-20260908/build/lib.linux-x86_64-cpython-311/pymica/pymica.cpython-311-x86_64-linux-gnu.so')
ic=(ct.c_int*1)(0);il0=(ct.c_int*1)(3);il=(ct.POINTER(ct.c_int)*1)(il0)
ct.c_void_p.in_dll(lib,'idx_con_pm').value=ct.addressof(ic)
ct.c_void_p.in_dll(lib,'idx_line_pm').value=ct.addressof(il)
ct.c_int.in_dll(lib,'num_gaussian').value=2
Pdouble=ct.POINTER(ct.c_double)
pt=lambda a:a.ctypes.data_as(Pdouble)
for suffix in ['gauss','gamma']:
 f=getattr(lib,'Slc_array_'+suffix);f.argtypes=[Pdouble,ct.c_int,Pdouble,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,Pdouble];f.restype=None
 f=getattr(lib,'Sll_array_'+suffix);f.argtypes=[Pdouble,ct.c_int,ct.c_void_p,ct.c_int,ct.c_int,Pdouble];f.restype=None
fft_modelcov=modelcov
def modelcov(q,kind):
    sig,tau=np.exp(q[:2])
    pm=np.array([0,q[0]-.5*q[1],q[1],0,q[2],q[3],q[4]-(.5*np.log(2) if kind=='gamma' else 0),q[5],q[6],q[7]-(.5*np.log(2) if kind=='gamma' else 0)])
    xy=np.empty((nw,nu));yy=np.zeros((nu,nu));suffix='gauss' if kind=='gaussian' else 'gamma'
    getattr(lib,'Slc_array_'+suffix)(pt(tw),nw,pt(tu),nu,pt(pm),0,0,pt(xy))
    getattr(lib,'Sll_array_'+suffix)(pt(tu),nu,pt(pm),0,0,pt(yy))
    yy=yy+np.tril(yy,-1).T
    xx=np.exp(-np.abs(dw)/tau)
    return sig**2*np.block([[xx,xy],[xy.T,yy]])+noise

def empirical_approx(Ct,wp):
    cw=Ct[:nw,:nw]
    ones=np.ones((nw,1))
    def obj(z):
        sig,tau=np.exp(z)
        K=sig**2*np.exp(-np.abs(dw)/tau)+np.diag(ew**2)
        cf=cho_factor(K,lower=True,check_finite=False)
        iz=cho_solve(cf,ones,check_finite=False);B=(ones.T@iz).item()
        return .5*(2*np.log(np.diag(cf[0])).sum()+np.log(B)+np.trace(cho_solve(cf,cw,check_finite=False))-(iz.T@cw@iz).item()/B)
    hh=.01;Hess=np.zeros((2,2));eye=np.eye(2)
    for i in range(2):
        for j in range(2):
            Hess[i,j]=(obj(wp+hh*eye[i]+hh*eye[j])-obj(wp+hh*eye[i]-hh*eye[j])-obj(wp-hh*eye[i]+hh*eye[j])+obj(wp-hh*eye[i]-hh*eye[j]))/(4*hh*hh)
    T=np.array([[1,-.5],[0,1.]])
    cov=T@np.linalg.inv(Hess)@T.T
    means=T@wp;errs=np.sqrt(np.diag(cov))
    print('approx_cont_prior',means.tolist(),errs.tolist(),flush=True)
    return means,errs

from functools import lru_cache
from starkey_disk_transfer import disk_transfer_function, disk_moments
dt=.025;N=2**19
freq=2*np.pi*np.fft.rfftfreq(N,dt)
@lru_cache(maxsize=16)
def diskhat(T1,wave):
    fine=dt/4; count=4*N
    # Midpoint quadrature on a finite support, with explicit half-cell phase.
    # 100 days is much longer than these sub-day standard-disk kernels.
    td=(np.arange(int(100/fine))+.5)*fine
    den=disk_transfer_function(td,wavelength_rest_angstrom=wave/1.031,
        t1_kelvin=T1,inclination_deg=45,redshift=.031)
    g,w=np.polynomial.legendre.leggauss(24)
    cells=min(256,len(td))
    fine_points=td[:cells,None]+.5*fine*g
    den[:cells]=(disk_transfer_function(fine_points,
        wavelength_rest_angstrom=wave/1.031,t1_kelvin=T1,
        inclination_deg=45,redshift=.031)@w)/2
    area=den.sum()*fine
    if abs(area-1)>0.005: raise ValueError(('disk integration mass',area))
    H=np.fft.rfft(den,n=count)[:len(freq)]*fine*np.exp(-.5j*freq*fine)
    return H,area
def disk_true(p,verbose=False):
    T1=p.get('T1',5900.); tau=p.get('tauI',30.)
    q=p.get('q',.623041375227);Ts=p.get('Ts',55.3735295468)
    a=p.get('a',.787);b=p.get('b',.15);hg=p.get('hgain',1.27560869565)
    delta=p.get('delta',2.90964381881)
    DW,aw=diskhat(T1,1928.); DU,au=diskhat(T1,3465.)
    B=np.exp(-1j*freq*p.get('blr_onset',6.))/(1+1j*freq*p.get('blr_theta',7.))**2
    PI=2*tau/(1+(freq*tau)**2)
    Ps=4*Ts/(1+(freq*Ts)**2)**2
    v=p.get('v',0.);Tu=p.get('Tu',150.)
    Pu=4*Tu/(1+(freq*Tu)**2)**2
    # xW and xU share smooth s, but are independent of illuminating I0.
    # Optional xU-only independent u; all PSD matrices are PSD by construction.
    XW=DW
    XU=DU*np.exp(-1j*freq*delta)
    V=a*DU+b*B
    ww=np.abs(DW)**2*PI+q*np.abs(XW)**2*Ps
    uw=V*np.conj(DW)*PI+hg*q*XU*np.conj(XW)*Ps
    uu=np.abs(V)**2*PI+hg**2*q*np.abs(XU)**2*Ps+v*Pu
    q2=p.get('q2',0.);Ts2=p.get('Ts2',10.);h2=p.get('h2',1.1);delta2=p.get('delta2',30.)
    P2=4*Ts2/(1+(freq*Ts2)**2)**2
    V2=DU*np.exp(-1j*freq*delta2)
    ww+=q2*abs(DW)**2*P2
    uw+=q2*h2*V2*np.conj(DW)*P2
    uu+=q2*h2*h2*abs(V2)**2*P2
    xx=fft_cov(ww,dw);xy=fft_cov(uw,h);yy=fft_cov(uu,du)
    dv=lambda z:np.trace(z)/len(z)-np.mean(z)
    alpha=.16900624325231342/np.sqrt(dv(xx))
    Ct=alpha**2*np.block([[xx,xy],[xy.T,yy]])+noise
    if verbose:
        print('TRUE',json.dumps(dict(**p,alpha=alpha,Wrms=.16900624325231342,
          Urms=alpha*np.sqrt(dv(yy)),disk_areas=[aw,au],
          disk_moments=[disk_moments(wavelength_rest_angstrom=w/1.031,
           t1_kelvin=T1,inclination_deg=45,redshift=.031) for w in (1928,3465)])),flush=True)
    return Ct
def w_profile(Ct):
    cw=Ct[:nw,:nw]; ones=np.ones((nw,1))
    def obj(z):
        sig,tau=np.exp(z);K=sig**2*np.exp(-np.abs(dw)/tau)+np.diag(ew**2)
        cf=cho_factor(K,lower=True,check_finite=False)
        iz=cho_solve(cf,ones,check_finite=False);B=(ones.T@iz).item()
        return .5*(2*np.log(np.diag(cf[0])).sum()+np.log(B)+np.trace(cho_solve(cf,cw,check_finite=False))-(iz.T@cw@iz).item()/B)
    opt=minimize(obj,np.log([.2,50]),method='L-BFGS-B',bounds=[(np.log(.03),np.log(2)),(np.log(1),np.log(5000))],options={'ftol':1e-10})
    return opt.x
def fitted(Ct,kinds=('gaussian','gamma'), maxiter=110, extra_starts=None, wide_narrow=False):
    wp=w_profile(Ct);means,errs=empirical_approx(Ct,wp);rows=[]
    for kind in kinds:
        bounds=getbounds(kind)
        if wide_narrow:
            bounds[2]=(np.log(1e-5),np.log(10));bounds[3]=(0,100);bounds[4]=bounds[7]
        initial=[qpars(*np.exp(wp),.787,.204,.21 if kind=='gaussian' else .34,.386,11.26,29.08),
                 qpars(*np.exp(wp),.84,.2,.21 if kind=='gaussian' else .34,5,4,150 if kind=='gaussian' else 214)]
        if extra_starts: initial+=extra_starts.get(kind,[])
        def objective(q):
            if kind=='gaussian' and q[3]>q[6]: return 1e8+(q[3]-q[6])*1e6
            try: value=loss(modelcov(q,kind),Ct)
            except (np.linalg.LinAlgError,ValueError):return 1e50
            zz=np.array([q[0]-.5*q[1],q[1]])
            if np.any(abs(zz-means)>5*errs): return value+1e5+sum((zz-means)**2)
            return value+.5*np.sum(((zz-means)/errs)**2)
        for j,q0 in enumerate(initial):
            begin=time.time()
            opt=minimize(objective,q0,method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-9,'maxls':20,'gtol':1e-4})
            row=dict(kind=kind,start=j,objective=float(opt.fun),likelihood_cost=loss(modelcov(opt.x,kind),Ct),
              success=bool(opt.success),message=str(opt.message),nfev=opt.nfev,seconds=time.time()-begin,q=opt.x.tolist(),**reportq(opt.x))
            print('FIT',json.dumps(row),flush=True);rows.append(row)
    return rows

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description='Expected native-MICA likelihood projection; no MICA sampling.')
    ap.add_argument('--params',default='{}',help='JSON overrides of generator parameters')
    ap.add_argument('--case',help='Case id in adjacent prediction_candidates.json')
    ap.add_argument('--maxiter',type=int,default=110)
    ap.add_argument('--full-bounds',action='store_true')
    args=ap.parse_args()
    p={}
    if args.case:
        saved=json.loads(Path(__file__).with_name('prediction_candidates.json').read_text())
        p=dict(next(c['params'] for c in saved['cases'] if c['id']==args.case))
    p.update(json.loads(args.params))
    Ct=disk_true(p,True)
    fitted(Ct,maxiter=args.maxiter,wide_narrow=args.full_bounds)
