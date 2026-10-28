"""Exploratory score-matching design, then full-timestamp local validation.
This is the second attempted tuning round, NOT a successful calibrated design.
Outputs go to stdout; does not launch MICA or alter existing artifacts.
"""
from predict_disk import *


from scipy.optimize import least_squares
qG=qpars(.196,38.5,.787,.204,.14195358,.386,11.26,29.08)
qY=qpars(.224,51.1,.833,.167,.200752678,7.137,0.,216.148400)
pinit=dict(T1=1500.,tauI=30.,q=.294,Ts=89.,a=.831,b=.175,hgain=1.388,delta=.01,q2=.025,Ts2=10.,h2=.8,delta2=30.)
Ct0=disk_true(pinit);wp=w_profile(Ct0);means,errs=empirical_approx(Ct0,wp)
def prec(C):
 cf=cho_factor(C,lower=True,check_finite=False);iv=cho_solve(cf,np.eye(n),check_finite=False)
 iL=iv@L;Z=L.T@iL
 return iv-iL@np.linalg.solve(Z,iL.T)
mats=[];const=[];scales=[];direction=[]
for kind,q0 in [('gaussian',qG),('gamma',qY)]:
 C=modelcov(q0,kind);P=prec(C)
 zz=np.array([q0[0]-.5*q0[1],q0[1]])
 gp=np.array([[1,0],[-.5,1.]])@((zz-means)/errs**2)
 for k in range(8):
  dq=np.zeros(8);dq[k]=.002 if k not in (3,6) else .02
  dC=(modelcov(q0+dq,kind)-modelcov(q0-dq,kind))/(2*dq[k])
  mats.append(-.5*P@dC@P)
  const.append(.5*np.sum(P*dC)+(gp[k] if k<2 else 0))
  scales.append(np.sqrt(max(.5*np.trace(P@dC@P@dC),1e-8)))
  # lower boundary requires nonnegative score; upper requires nonpositive.
  direction.append(1 if k==4 or (kind=='gamma' and k==6) else -1 if kind=='gamma' and k==7 else 0)
mats=np.asarray(mats);const=np.array(const);scales=np.array(scales)
keys=['q','Ts','hgain','delta','a','b','q2','Ts2','h2','delta2']
logkeys={'q','Ts','hgain','a','b','q2','Ts2','h2'}
lo=dict(q=.1,Ts=25,hgain=1.001,delta=0,a=.7,b=.05,q2=.002,Ts2=5,h2=.2,delta2=5)
hi=dict(q=3.,Ts=200,hgain=1.8,delta=10,a=.95,b=.35,q2=.3,Ts2=30,h2=2.5,delta2=60)
pack=lambda p:np.array([np.log(p[k]) if k in logkeys else p[k] for k in keys])
def unpack(x):
 p=dict(pinit)
 for k,v in zip(keys,x):p[k]=float(np.exp(v) if k in logkeys else v)
 return p
counter=[0];best=[1e100,None]
def residual(x):
 p=unpack(x);Ct=disk_true(p)
 g=(const+np.einsum('kij,ij->k',mats,Ct))/scales
 for k,d in enumerate(direction):
  if d==1:g[k]=min(g[k],0)
  elif d==-1:g[k]=max(g[k],0)
 # nuisance gradients soft constraints, narrow shape near lower bound unconstrained
 g[[0,1,8,9]]*=.3
 g[[3,11]]*=1.5
 g[13]*=6.
 g[[5,6,7]]*=2.
 yy=Ct[nw:,nw:]-noise[nw:,nw:]
 urms=np.sqrt(np.trace(yy)/nu-np.mean(yy))
 g=np.r_[g,(urms-.17052735)/.03]
 val=float(g@g);counter[0]+=1
 if val<best[0]:best[:]=val,p
 if counter[0]%50==1: print('SCORE_PROGRESS',counter[0],val,json.dumps(p),flush=True)
 return g
opt=least_squares(residual,pack(pinit),bounds=(pack(lo),pack(hi)),max_nfev=35,ftol=1e-5,xtol=1e-5,gtol=1e-5,diff_step=1e-4)
p=unpack(opt.x);print('SCORE_OPTIMUM',json.dumps(dict(params=p,score=float(opt.fun@opt.fun),nfev=opt.nfev,success=bool(opt.success))),flush=True)
Ct=disk_true(p,True);fitted(Ct,maxiter=110,wide_narrow=True)

