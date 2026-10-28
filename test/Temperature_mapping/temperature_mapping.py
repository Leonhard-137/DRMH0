"""NK22 linear temperature inversion in fractional-temperature coordinates.

Run with the existing astro Python. All fluxes stay in observed 1e-15
erg/s/cm2/Angstrom and times in observed THJD; the forward kernel includes z.
"""
from pathlib import Path
import argparse, json, time, gc
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
C = 2.99792458e10
G = 6.67430e-8
H = 6.62607015e-27
KB = 1.380649e-16
SIGMA = 5.670374419e-5
MSUN = 1.98847e33
MPC = 3.085677581491367e24
DAY = 86400.0
DEFAULT = dict(nr=50, nt=250, mass_msun=3.85e7, distance_mpc=136.,
               redshift=.031455, eddington_ratio=.2, efficiency=.1,
               inclination_deg=30., rin_rg=6., umax=1000.,
               radial_quadrature=128, time_boundary="renormalized_hats",
               flux_unit="1e-15 erg s^-1 cm^-2 Angstrom^-1",
               time_unit="observer THJD = HJD - 2450000",
               radial_penalty="NK22 absolute deltaT difference / geometric T",
               smoothing="linear interpolate to 0.1-day grid, Gaussian, constant endpoints")

def disk_temperature(u, p):
    mass = p["mass_msun"] * MSUN
    rin = p["rin_rg"] * G * mass / C**2
    ledd = 1.26e38 * p["mass_msun"]
    mdot = ledd*p["eddington_ratio"]/(p["efficiency"]*C**2)
    tin = (3*G*mass*mdot/(8*np.pi*SIGMA*rin**3))**.25
    u = np.asarray(u)
    return tin*u**(-.75)*np.maximum(1-u**(-.5), 0)**.25, rin, mdot

def radial_kernels(wavelengths, p):
    edges = np.geomspace(p.get("umin", 1.), p["umax"], p["nr"]+1)
    u = np.sqrt(edges[:-1]*edges[1:])
    temp, rin, mdot = disk_temperature(u, p)
    lam = np.asarray(wavelengths)/(1+p["redshift"])*1e-8
    f0 = 4*np.pi*H*C*C*rin**2*np.cos(np.deg2rad(p["inclination_deg"]))
    f0 = f0/(lam**5*(p["distance_mpc"]*MPC)**2) * 1e7/(1+p["redshift"])
    kernel = np.zeros((len(lam),len(u)))
    for k in range(len(u)):
        ur = np.geomspace(edges[k],edges[k+1],p["radial_quadrature"]+1)
        tr = disk_temperature(ur,p)[0]
        x = H*C/(lam[:,None]*KB*np.maximum(tr[None,:],1e-30))
        xx = np.minimum(x, 740)
        v = xx*np.exp(-xx)/np.maximum(-np.expm1(-xx),1e-200)**2
        integrand = ur[None,:]*v/np.maximum(tr[None,:],1e-30)
        kernel[:,k] = f0*trapezoid(integrand,ur,axis=1)*temp[k]
    return u, edges, temp, rin, kernel

def temporal_weights(td, tp, t0, normalize=True):
    """Analytic integral of triangular interpolation basis over an inclined ring."""
    dt = tp[1]-tp[0]
    d = np.asarray(td)[:,None]-tp[None,:]
    if t0 <= 1e-12*dt:
        f = np.maximum(0., 1-np.abs(d)/dt)
        if normalize:
            f /= f.sum(axis=1,keepdims=True)
        return f
    def terms(a,b):
        aa=np.clip(a,-t0,t0); bb=np.clip(b,-t0,t0)
        prob=(np.arcsin(bb/t0)-np.arcsin(aa/t0))/np.pi
        moment=(np.sqrt(np.maximum(t0*t0-aa*aa,0))-
                np.sqrt(np.maximum(t0*t0-bb*bb,0)))/np.pi
        return prob,moment
    pl,ml=terms(d-dt,d)
    pr,mr=terms(d,d+dt)
    f=((dt-d)*pl+ml+(dt+d)*pr-mr)/dt
    f=np.maximum(f,0)
    f[f<1e-13]=0
    if normalize:
        f /= f.sum(axis=1,keepdims=True)
    return f

def regularizer(temp, nt):
    nr=len(temp); n=nr*nt
    # Time-major ordering permits sparse local temporal coupling.
    idx=np.arange(n).reshape(nt,nr)
    rows=np.repeat(np.arange(nt*(nr-1)),2)
    cols=np.stack((idx[:,:-1].ravel(),idx[:,1:].ravel()),axis=1).ravel()
    ratio=np.tile(temp[1:]/temp[:-1],nt)
    val=np.stack((-1/np.sqrt(ratio),np.sqrt(ratio)),axis=1).ravel()
    dr=sparse.csr_matrix((val,(rows,cols)),shape=(nt*(nr-1),n))
    rows=np.repeat(np.arange((nt-1)*nr),2)
    cols=np.stack((idx[:-1].ravel(),idx[1:].ravel()),axis=1).ravel()
    val=np.tile([-1.,1.],(nt-1)*nr)
    dt=sparse.csr_matrix((val,(rows,cols)),shape=((nt-1)*nr,n))
    return (sparse.eye(n,format="csc")+dr.T@dr+dt.T@dt).tocsc()

class Inversion:
    def __init__(self, t, band_index, wavelengths, error, params=None):
        self.p=DEFAULT.copy()
        if params: self.p.update(params)
        p=self.p; self.t=np.asarray(t); self.bi=np.asarray(band_index)
        self.error=np.asarray(error); self.nb=len(wavelengths)
        # Optional fixed bounds allow band-removal tests on exactly the same
        # temporal grid. Historical runs keep their original defaults.
        self.tp=np.linspace(p.get("time_grid_min", min(t)),
                            p.get("time_grid_max", max(t)),p["nt"])
        self.u,self.edges,self.temp,self.rin,self.kernel=radial_kernels(wavelengths,p)
        nr=p["nr"]; nt=p["nt"]
        vals=[]; rows=[]; cols=[]
        for k in range(nr):
            t0=self.u[k]*self.rin/C/DAY*np.sin(np.deg2rad(p["inclination_deg"]))*(1+p["redshift"])
            f=temporal_weights(t,self.tp,t0,p["time_boundary"]!="zero")
            rr,cc=np.nonzero(f)
            rows.append(rr); cols.append(cc*nr+k)
            vals.append(f[rr,cc]*self.kernel[self.bi[rr],k])
        self.A=sparse.csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                                 shape=(len(t),nr*nt))
        self.B=sparse.csr_matrix((np.ones(len(t)),(np.arange(len(t)),self.bi)),
                                shape=(len(t),self.nb))
        self.Aw=self.A.multiply((1/self.error)[:,None]).tocsr()
        self.Bw=self.B.multiply((1/self.error)[:,None]).tocsr()
        self.normal=(self.Aw.T@self.Aw).tocsc()
        self.cross=(self.Aw.T@self.Bw).toarray()
        self.bd=np.asarray(self.Bw.power(2).sum(axis=0)).ravel()
        self.P=regularizer(self.temp,nt)
        self.factor=None
    def factorize(self,xi):
        self.xi=float(xi)
        ts=time.perf_counter()
        self.factor=splu(self.normal+xi*self.P,permc_spec="MMD_AT_PLUS_A")
        self.Z=self.factor.solve(self.cross)
        self.S=np.diag(self.bd)-self.cross.T@self.Z
        self.S=(self.S+self.S.T)/2
        return time.perf_counter()-ts
    def solve_rhs(self, rhsq, rhsb):
        y=self.factor.solve(np.asarray(rhsq))
        b=np.linalg.solve(self.S,rhsb-self.cross.T@y)
        q=y-self.Z@b
        return q,b
    def solve(self,flux):
        fw=np.asarray(flux)/self.error
        q,b=self.solve_rhs(self.Aw.T@fw,self.Bw.T@fw)
        pred=self.A@q+self.B@b
        residual=np.asarray(flux)-pred
        # Verify the actual normal equations, including unpenalized band offsets.
        grad=self.Aw.T@(residual/self.error)-self.xi*self.P@q
        rel=np.linalg.norm(grad)/max(np.linalg.norm(self.Aw.T@fw),1.)
        return dict(q=q.reshape(self.p["nt"],self.p["nr"]).T,
                    baseline=b,pred=pred,residual=residual,
                    chi2=float(np.sum((residual/self.error)**2)),
                    chi2_per_point=float(np.mean((residual/self.error)**2)),
                    scale99=float(np.quantile(np.abs(q),.99)),
                    normal_equation_relative_residual=float(rel),
                    penalty=float(q@(self.P@q)))
    def edf(self,probes=48,seed=817):
        # Hutchinson trace estimator of the full weighted-data hat matrix.
        rng=np.random.default_rng(seed)
        v=rng.choice([-1.,1.],size=(len(self.t),probes))
        q,b=self.solve_rhs(self.Aw.T@v,self.Bw.T@v)
        hv=self.Aw@q+self.Bw@b
        sample=np.sum(v*hv,axis=0)
        return float(sample.mean()),float(sample.std(ddof=1)/np.sqrt(probes))

def smooth_values(t, y, fwhm, dt=.1):
    grid=np.arange(min(t),max(t)+dt/2,dt)
    yg=np.interp(grid,t,y)
    sm=gaussian_filter1d(yg,fwhm/(2*np.sqrt(2*np.log(2)))/dt,mode="nearest",truncate=5)
    return np.interp(t,grid,sm)

def data_variants(t,bi,flux,bands):
    variants={"baseline":flux.copy()}
    for width in [7,28,56]:
        sm=np.empty_like(flux)
        for b in range(len(bands)):
            ii=np.flatnonzero(bi==b)
            sm[ii]=smooth_values(t[ii],flux[ii],width)
        variants[f"smooth_{width}"]=sm
        variants[f"subtract_{width}"]=flux-sm
    injected=np.zeros_like(flux)
    for name in ["Swift_U","Ground_u"]:
        b=bands.index(name); ii=np.flatnonzero(bi==b)
        sm=smooth_values(t[ii],flux[ii],10.5)
        sm-=np.mean(sm)
        injected[ii]=np.interp(t[ii]-21,t[ii],sm)
    for frac in [.1,.2,.5]:
        variants["blr_"+str(frac).replace(".","p")]=flux+frac*injected
    return variants,injected

def load_data(path):
    data=np.genfromtxt(path,delimiter=",",names=True,dtype=None,encoding="utf8")
    bandlist=list(dict.fromkeys(data["band"]))
    wave=np.array([data["wavelength_obs_A"][data["band"]==b][0] for b in bandlist])
    order=np.argsort(wave); bands=[bandlist[i] for i in order]; wave=wave[order]
    bi=np.array([bands.index(x) for x in data["band"]])
    return data,bands,bi,wave

def run(args):
    threadpool_limits(limits=2)
    out=ROOT/"outputs";out.mkdir(exist_ok=True)
    data,bands,bi,wave=load_data(ROOT/"derived/lightcurves.csv")
    t=data["thjd"].astype(float);f=data["flux"].astype(float);e=data["error"].astype(float)
    params=DEFAULT.copy()
    if args.quick: params.update(nr=25,nt=100)
    start=time.perf_counter()
    model=Inversion(t,bi,wave,e,params)
    print(f"Forward matrix {model.A.shape}, nnz={model.A.nnz}, construction={time.perf_counter()-start:.2f}s",flush=True)
    variants,injection=data_variants(t,bi,f,bands)
    saved=dict(t=t,band=data["band"],flux=f,error=e,error_original=data["error_original"],
               bi=bi,bands=np.array(bands),wavelengths=wave,tp=model.tp,u=model.u,
               edges=model.edges,temp=model.temp,kernel=model.kernel,blr_template=injection)
    metrics=[]
    # Fixed logarithmic scan includes the four published regularizations.
    xis=[1.,3.,10.,30.,100.,300.,500.,1000.,3000.,10000.]
    if args.quick: xis=[1.,10.,100.,1000.]
    results={}
    for xi in xis:
        sec=model.factorize(xi)
        fit=model.solve(f); edf,edfse=model.edf(24 if args.quick else 64)
        name=f"xi_{xi:g}";results[name]=fit
        saved["q_"+name]=fit["q"]; saved["pred_"+name]=fit["pred"]
        metrics.append(dict(name=name,xi=xi,**{k:fit[k] for k in
            ["chi2","chi2_per_point","scale99","penalty","normal_equation_relative_residual"]},
            effective_parameters=edf,effective_parameters_mcse=edfse,
            aic=fit["chi2"]+2*edf,bic=fit["chi2"]+np.log(len(t))*edf,
            factor_seconds=sec))
        print(name,metrics[-1],flush=True)
        if xi==10:
            for key,y in variants.items():
                fit=model.solve(y);results[key]=fit
                saved["q_"+key]=fit["q"];saved["pred_"+key]=fit["pred"];saved["flux_"+key]=y
                metrics.append(dict(name=key,xi=10,**{k:fit[k] for k in
                    ["chi2","chi2_per_point","scale99","penalty","normal_equation_relative_residual"]}))
            # Exact linearity check for the low-pass/high-pass decomposition.
            smooth=results["smooth_56"];sub=results["subtract_56"];base=results["baseline"]
            linearity=float(np.max(np.abs(smooth["q"]+sub["q"]-base["q"])))
            print("linearity max abs q",linearity,flush=True)
    suffix="_quick" if args.quick else ""
    np.savez_compressed(out/f"fits{suffix}.npz",**saved)
    config=dict(params,band_names=bands,wavelengths_observer_A=wave.tolist(),
        n_data=len(t),tmin=float(min(t)),tmax=float(max(t)),elapsed_seconds=time.perf_counter()-start,
        edf_method="64 Rademacher probes, seed 817; MC standard error reported",
        linearity_max_abs_q=linearity)
    (out/f"config{suffix}.json").write_text(json.dumps(config,indent=2))
    (out/f"metrics{suffix}.json").write_text(json.dumps(metrics,indent=2))
    print("Saved",out,"total seconds",config["elapsed_seconds"],flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick",action="store_true")
    run(parser.parse_args())
