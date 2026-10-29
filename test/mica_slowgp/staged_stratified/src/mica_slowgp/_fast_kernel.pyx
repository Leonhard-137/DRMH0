# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Isolated scalar covariance assembly using SciPy's own special functions."""
import numpy as np
from libc.math cimport exp, fabs, hypot, isinf, log1p, sqrt
from scipy.special.cython_special cimport erfcx, ndtr


cdef double INV_SQRT_TWO = 0.707106781186547524400844362104849039
cdef double SQRT_TWO_OVER_PI = 0.797884560802865355879892119868763737


cdef inline double gaussian_ou(double delta, double width, double tau) noexcept nogil:
    cdef double d = fabs(delta)
    cdef double r, z, envelope, answer, exponent
    if width == 0.0:
        return exp(-d / tau)
    r = width / tau
    if isinf(r):
        z = d / width
        return SQRT_TWO_OVER_PI * (tau / width) * exp(-0.5 * z * z)
    if r == 0.0:
        return exp(-d / tau)
    z = d / width
    envelope = 0.5 * exp(-0.5 * z * z)
    answer = envelope * erfcx((r + z) * INV_SQRT_TWO)
    if z <= r:
        answer += envelope * erfcx((r - z) * INV_SQRT_TWO)
    else:
        if r <= 1.0:
            exponent = 0.5 * r * r - d / tau
        else:
            exponent = r * (0.5 * r - z)
        answer += exp(exponent) * ndtr(z - r)
    return answer


cdef inline double slow_kernel(double delta, double ts) noexcept nogil:
    cdef double r = fabs(delta) / ts
    if isinf(r):
        return 0.0
    return exp(log1p(r) - r)


def covariance(const double[:, ::1] dww, const double[:, ::1] dwb,
               const double[:, ::1] dbb, const double[::1] noise,
               const double[::1] p, int mode):
    """Build full symmetric covariance; validated inputs come from fast.py.

    p follows sigma_x,tau_x,A0,mu0,w0,A1,mu1,w1,qW,qb,Ts.
    mode: 0=none, 1=independent, 2=shared. This is private, not a validator.
    """
    cdef Py_ssize_t nw=dww.shape[0], nb=dbb.shape[0], i, j
    cdef double sigma2=p[0]*p[0], tau=p[1]
    cdef double a0=p[2], mu0=p[3], w0=p[4], a1=p[5], mu1=p[6], w1=p[7]
    cdef double qw=p[8], qb=p[9], ts=p[10]
    cdef double s00=sigma2*a0*a0, s01=sigma2*a0*a1
    cdef double s10=sigma2*a1*a0, s11=sigma2*a1*a1
    cdef double w00=hypot(w0,w0), w01=hypot(w0,w1), w11=hypot(w1,w1)
    cdef double delta, value
    array = np.empty((nw+nb, nw+nb), dtype=np.float64, order="F")
    cdef double[::1, :] out = array
    with nogil:
        for i in range(nw):
            for j in range(i+1):
                delta = dww[i,j]
                value = sigma2 * exp(-fabs(delta)/tau)
                if mode != 0 and qw != 0.0:
                    value += qw*qw*slow_kernel(delta,ts)
                out[i,j] = value
                out[j,i] = value
        for i in range(nw):
            for j in range(nb):
                delta = dwb[i,j]
                value = 0.0
                if a0 != 0.0:
                    value += sigma2*a0*gaussian_ou(delta+mu0,w0,tau)
                if a1 != 0.0:
                    value += sigma2*a1*gaussian_ou(delta+mu1,w1,tau)
                if mode == 2 and qw != 0.0 and qb != 0.0:
                    value += qw*qb*slow_kernel(delta,ts)
                out[i,nw+j] = value
                out[nw+j,i] = value
        for i in range(nb):
            for j in range(i+1):
                delta = dbb[i,j]
                value = 0.0
                if a0 != 0.0:
                    value += s00*gaussian_ou(delta,w00,tau)
                if a0 != 0.0 and a1 != 0.0:
                    value += s01*gaussian_ou(delta-mu0+mu1,w01,tau)
                    value += s10*gaussian_ou(delta-mu1+mu0,w01,tau)
                if a1 != 0.0:
                    value += s11*gaussian_ou(delta,w11,tau)
                if mode != 0 and qb != 0.0:
                    value += qb*qb*slow_kernel(delta,ts)
                out[nw+i,nw+j] = value
                out[nw+j,nw+i] = value
        for i in range(nw+nb):
            out[i,i] += noise[i]
    return array


def gaussian_ou_values(const double[::1] delta, double width, double tau):
    """Private direct kernel comparison hook (caller supplies valid scalars)."""
    cdef Py_ssize_t i
    result = np.empty(delta.shape[0], dtype=np.float64)
    cdef double[::1] out = result
    with nogil:
        for i in range(delta.shape[0]):
            out[i] = gaussian_ou(delta[i],width,tau)
    return result
