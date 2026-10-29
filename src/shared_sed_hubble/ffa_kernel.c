/* Exact scalar semiseparable FFA kernel, ported from FFA/ffa/ffa.py.
 * C = diag(err^2) + R_band(i) R_band(j) exp(-|t_i-t_j|/tau).
 * Offsets A have the same flat-prior Gaussian marginalization as upstream.
 * No likelihood/noise-model changes, priors, or parameter restrictions here.
 */
#include <math.h>
#include <stddef.h>

/* 求解 z=C^-1 rhs：利用指数核的半可分结构，前后各扫一次观测。
 * 这里不构造 N×N 稠密逆矩阵；U、D、W、phi 来自下方因子分解。 */
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

/* 已知小矩阵 Q=L L^T 的 Cholesky 因子，通过两次三角回代求 Q^-1 rhs。
 * 此处的 L 是 Cholesky 因子，区别于正文中表示波段归属的设计矩阵。 */
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
/* 输入约定：t 已按时间排序，band 从 0 编号；coords=(ln tau, ln R_0,...)。
 * 各波段常数背景 A 用平坦先验解析边缘化，采样器只需处理协方差参数。
 * 返回码 1/2/3/4/5/6 分别对应维数、参数、波段索引、C 分解、Q 分解、似然异常。 */
int ffa_evaluate(int n, int p, const double *t, const double *y,
                 const double *err, const int *band, const double *coords,
                 double *work, double *stats, double *mean, double *cov,
                 double *invres, int details) {
    stats[0]=-INFINITY; stats[1]=INFINITY;
    if(n<1 || p<1) return 1;
    /* 连续工作区被切为固定大小的视图；调用者负责分配足够空间。
     * 复用这些内存避免每次似然求值 malloc，也意味着实例不能被线程同时调用。 */
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
    /* 指数相关在相邻时刻间的传播为 phi_i=exp[-(t_i-t_{i-1})/tau]。
     * D 是半可分分解的对角枢轴；sum(log D_i) 给出 log det C。 */
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
    /* 设计矩阵的第 j 列在 band==j 处为 1，其余为 0。
     * 缓存 C^-1 L 后，只需在波段数 p 的小矩阵上处理常数背景。 */
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
    /* Q=L^T C^-1 L，rhsq=L^T C^-1 y。
     * 对称化抑制舍入误差；Cholesky 同时得到 log det Q 和背景条件均值所需因子。 */
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
    /* Amean=Q^-1 rhsq；以 y-L Amean 计算广义最小二乘残差的二次型。 */
    chol_solve(p,chol,rhsq,mean);
    for(int i=0;i<n;i++) rhs[i]=y[i]-mean[band[i]];
    inv_vector(n,U,D,W,phi,rhs,invres);
    double quad=0.0;
    for(int i=0;i<n;i++) quad+=rhs[i]*invres[i];
    stats[1]=quad;
    /* log det Q 是边缘化常数背景的归一化项，不能在参数比较时省略。
     * 与协方差参数无关的加性常数被省略，因此这里不是完整归一化证据值。 */
    stats[0]=-0.5*(quad+logdet+logdetq);
    if(!isfinite(stats[0])) return 6;
    /* 需要背景的不确定度时逐列求 Q^-1；loglike 路径省去这一步。
     * 此 cov 是给定当前 tau、R 的条件协方差，不是把所有后验样本混合后的总协方差。 */
    if(details) {
        for(int j=0;j<p;j++) {
            for(int i=0;i<p;i++) unit[i]=(i==j ? 1.0 : 0.0);
            chol_solve(p,chol,unit,sol);
            for(int i=0;i<p;i++) cov[i*p+j]=sol[i];
        }
    }
    return 0;
}
