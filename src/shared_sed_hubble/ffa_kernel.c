/* Exact scalar semiseparable FFA kernel, ported from FFA/ffa/ffa.py.
 * C = diag(err^2) + R_band(i) R_band(j) exp(-|t_i-t_j|/tau).
 * Offsets A have the same flat-prior Gaussian marginalization as upstream.
 * No likelihood/noise-model changes, priors, or parameter restrictions here.
 */
#include <math.h>
#include <stddef.h>

static void inv_vector(int n, const double *U, const double *D,
                       const double *W, const double *phi,
                       const double *rhs, double *z) {
    double f=0.0, g=0.0;
    z[0]=rhs[0];
    for(int i=1;i<n;i++) {
        f=phi[i]*(f+W[i-1]*z[i-1]);
        z[i]=rhs[i]-U[i]*f;
    }
    z[n-1]/=D[n-1];
    for(int i=n-2;i>=0;i--) {
        g=phi[i+1]*(g+U[i+1]*z[i+1]);
        z[i]=z[i]/D[i]-W[i]*g;
    }
}

static void chol_solve(int p, const double *L, const double *rhs, double *v) {
    for(int i=0;i<p;i++) {
        double a=rhs[i];
        for(int j=0;j<i;j++) a-=L[i*p+j]*v[j];
        v[i]=a/L[i*p+i];
    }
    for(int i=p-1;i>=0;i--) {
        double a=v[i];
        for(int j=i+1;j<p;j++) a-=L[j*p+i]*v[j];
        v[i]=a/L[i*p+i];
    }
}

/* work has n*(p+5)+2*p*p+3*p doubles; all arrays are C-contiguous.
 * stats = (loglike, quadratic form); invres = C^-1(y-L Amean).
 * details=0 omits cov only; Amean and invres are always returned.
 */
int ffa_evaluate(int n, int p, const double *t, const double *y,
                 const double *err, const int *band, const double *coords,
                 double *work, double *stats, double *mean, double *cov,
                 double *invres, int details) {
    stats[0]=-INFINITY; stats[1]=INFINITY;
    if(n<1 || p<1) return 1;
    double *U=work, *D=U+n, *W=D+n, *phi=W+n, *CL=phi+n;
    double *rhs=CL+n*p, *Cq=rhs+n, *chol=Cq+p*p;
    double *rhsq=chol+p*p, *unit=rhsq+p, *sol=unit+p;
    const double tau=exp(coords[0]);
    if(!(tau>0.0) || !isfinite(tau)) return 2;
    for(int i=0;i<n;i++) {
        if(band[i]<0 || band[i]>=p) return 3;
        U[i]=exp(coords[1+band[i]]);
        if(!isfinite(U[i])) return 2;
    }
    phi[0]=0.0;
    double S=0.0, logdet=0.0;
    D[0]=err[0]*err[0]+U[0]*U[0];
    if(!(D[0]>0.0) || !isfinite(D[0])) return 4;
    W[0]=U[0]/D[0]; logdet=log(D[0]);
    for(int i=1;i<n;i++) {
        phi[i]=exp(-(t[i]-t[i-1])/tau);
        S=phi[i]*phi[i]*(S+D[i-1]*W[i-1]*W[i-1]);
        D[i]=(err[i]*err[i]+U[i]*U[i])-U[i]*U[i]*S;
        if(!(D[i]>0.0) || !isfinite(D[i])) return 4;
        W[i]=(U[i]-U[i]*S)/D[i];
        logdet+=log(D[i]);
    }
    /* Store each C^-1 L column contiguously. */
    for(int j=0;j<p;j++) {
        for(int i=0;i<n;i++) rhs[i]=(band[i]==j ? 1.0 : 0.0);
        inv_vector(n,U,D,W,phi,rhs,CL+j*n);
    }
    for(int i=0;i<p*p;i++) Cq[i]=chol[i]=0.0;
    for(int j=0;j<p;j++) {
        rhsq[j]=0.0;
        for(int i=0;i<n;i++) {
            const double a=CL[j*n+i];
            rhsq[j]+=y[i]*a;
            Cq[band[i]*p+j]+=a;
        }
    }
    double logdetq=0.0;
    for(int i=0;i<p;i++) {
        for(int j=0;j<=i;j++) {
            double a=0.5*(Cq[i*p+j]+Cq[j*p+i]);
            for(int k=0;k<j;k++) a-=chol[i*p+k]*chol[j*p+k];
            if(i==j) {
                if(!(a>0.0) || !isfinite(a)) return 5;
                chol[i*p+j]=sqrt(a); logdetq+=log(a);
            } else chol[i*p+j]=a/chol[j*p+j];
        }
    }
    chol_solve(p,chol,rhsq,mean);
    for(int i=0;i<n;i++) rhs[i]=y[i]-mean[band[i]];
    inv_vector(n,U,D,W,phi,rhs,invres);
    double quad=0.0;
    for(int i=0;i<n;i++) quad+=rhs[i]*invres[i];
    stats[1]=quad;
    stats[0]=-0.5*(quad+logdet+logdetq);
    if(!isfinite(stats[0])) return 6;
    if(details) {
        for(int j=0;j<p;j++) {
            for(int i=0;i<p;i++) unit[i]=(i==j ? 1.0 : 0.0);
            chol_solve(p,chol,unit,sol);
            for(int i=0;i<p;i++) cov[i*p+j]=sol[i];
        }
    }
    return 0;
}
