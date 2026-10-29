# Independent variability and MICA: covariance calculation

This note records the analytic calculation accompanying the fixed-posterior
native influence measurements in README.md. No additional MICA fitting was
performed. The framework agrees with Li, Wang & Bai (2016), equations 3–16:
https://arxiv.org/pdf/1608.03741. The independent-signal extensions and the
finite-window example below are calculations for this analysis.

## Model and expected likelihood

Let latent driving history x have covariance K. The observed reference is
c=R x+n_c, and the true target is y=H_d x+u+n_y, with Cov(x,u)=0.
R and H include the actual observation times; x includes unobserved history.
Assume zero population mean for this covariance calculation. A nonzero
unmodelled mean adds its outer product to the expected quadratic term.
Independence alone specifies no timescale or stationarity for u.

MICA substitutes H=H_1+H_2 and omits u. Its joint covariance is

\[
C_\theta=
\begin{pmatrix}
RKR^T+N_c & RKH^T\\
HKR^T & HKH^T+N_y
\end{pmatrix}.
\]

The true independent signal adds only the target-target block:

\[
C_*=C_d+\begin{pmatrix}0&0\\0&U\end{pmatrix},
\qquad U=\operatorname{Cov}(u).
\]

For the two constant-mean columns L, set

\[
P_\theta=C_\theta^{-1}-C_\theta^{-1}L
(L^TC_\theta^{-1}L)^{-1}L^TC_\theta^{-1}.
\]

The expected negative marginalized log likelihood, up to a parameter-independent
constant, is

\[
J(\theta)=\tfrac12\{\log|C_\theta|
+\log|L^TC_\theta^{-1}L|+\operatorname{tr}(P_\theta C_*)\}.
\]

Parameter priors additionally affect the posterior. This is a specified way to
predict average model pressure once U is supplied; independence alone does not
specify U and cannot select a unique width or lag. The expected Gaussian
quadratic likelihood needs second moments, even when the underlying u is not
Gaussian. Equality of Gaussian laws from covariance equality additionally
requires the Gaussian assumption.

## Exact fixed-parameter reconstruction effect

Ignoring mean notation temporarily, condition the latent driver on the observed
reference: m=E[x|c], S=Cov(x|c). Put Q_jk=H_j S H_k^T, V=sum_jk Q_jk+N_y.
Then

\[
\widehat s_j=H_jm+W_j(y-Hm),\qquad
W_j=(Q_{j1}+Q_{j2})V^{-1}.
\]

At fixed theta, an added target-only vector u produces

\[
\Delta\widehat s_j=W_ju,\qquad
\operatorname{Cov}_u(\Delta\widehat s_j)=W_jUW_j^T.
\]

These identities do not require the particular vector u to be stationary,
Gaussian, or slow. The actual code uses the full mean-marginalized equivalent
D K_jz P_theta E_y u, with fixed native normalization and D removing the target
constant. It was checked by independent native full-minus-u reconstruction.
Parameter adaptation upon refitting is a separate effect; see
paired_short_effects.json.

If the complete noiseless driver history were known, S=0 and this target-update
route would vanish. Given only finite/noisy reference data, unobserved driver
freedom permits a nonzero W_j. When the second component uses nearly independent
remote driver history, Q_12 is small and Q_22 can resemble U. Its reconstructed
mean is then approximately U(Q_11+U+N_y)^{-1}(y-Hm), a smoothing of target
innovation, even though the code labels the component a response.

## A displaced Gamma kernel

For a DRW R_x(h)=sigma_x^2 exp(-|h|/d), define a shape-2 Gamma kernel

\[
\psi(s)=A(s-c)w^{-2}e^{-(s-c)/w}1_{s\ge c},
\qquad \tau=c+2w.
\]

For h<=c, direct integration gives

\[
R_{x\ell}(h)=A\sigma_x^2
\left({d\over d+w}\right)^2 e^{-(c-h)/d}.
\]

If the entire observing span is T and c>T, all observed cross covariances
are bounded by A sigma_x^2 [d/(d+w)]^2 exp[-(c-T)/d].
However, translating the kernel cancels out of its autocovariance:

\[
R_{\ell\ell}(h)=\iint\psi(s)\psi(r)R_x(h-s+r)\,ds\,dr.
\]

Its variance and two-sided angular-frequency PSD are

\[
R_{\ell\ell}(0)=\frac{A^2\sigma_x^2}{2}(q+q^2),\quad q={d\over d+w},
\qquad
P_\ell(\omega)=\frac{2A^2\sigma_x^2d}
{[1+(\omega d)^2][1+(\omega w)^2]^2}.
\]

Consequently, lag displacement controls coupling to the observed reference;
width controls the additional process's covariance shape. Nonresponse does not
imply a broad kernel. If U belongs to this filtered-DRW covariance family, a
sufficiently remote same-driver response approaches the independent Gaussian
model on any fixed finite observation set. Its cross covariance with the disk
also tends to zero. This is a finite-window identifiability limitation.

## Numerical existence example

finite_window_equivalence.py uses the actual disk_only W2/B observing times
(320 reference and 318 target samples, span 708.58664 d) and measurement errors.
It specifies an analytic toy, not the original thin-disk injection:

- DRW timescale 20 d and stationary sigma 0.2.
- Unit-gain delta disk response at 1 d.
- Independent target process with the same covariance as an independent DRW
  filtered by a Gamma kernel of width 20 d.
- Independent/disk stationary RMS ratio 0.3; Gamma area 0.4898979486.
- The competing response model retains the same disk, DRW and kernel width/area,
  changing only the Gamma onset c. This is an existence construction, not an
  optimization over all response parameters.
- Two orthogonal within-band mean contrasts remove the unknown constants,
  leaving a 636-dimensional likelihood.

For true covariance C_0 and response covariance C_r in this contrast space,

\[
D_{\rm KL}(0\Vert r)=\tfrac12[
\log|C_r|-\log|C_0|+\operatorname{tr}(C_r^{-1}C_0)-636].
\]

It is the true-model expectation of the log likelihood ratio, in nats, not
the observed likelihood ratio of the existing simulation realization and not
a calibrated model-selection detection threshold.

| Gamma onset c / d | Centroid / d | Expected log likelihood difference |
|---:|---:|---:|
| 0 | 40 | 215.552 |
| 400 | 440 | 19.4275 |
| 710 | 750 | 0.0391151 |
| 750 | 790 | 0.000643124 |
| 800 | 840 | 0.00000432389 |
| 900 | 940 | 0.000000000196297 |
| 1100 | 1140 | 4.04598e-19 |

At onset 1100 d the maximum driver-long correlation on any observed pair is
1.2926e-9, while the long component's entire autocovariance remains unchanged.
The covariance calculation and stable small-difference KL evaluation are saved
in finite_window_equivalence.json and the reproducible script above.

These two Gaussian models can therefore be observationally almost identical,
although one has a truly independent component and one has a common driver
over unobserved history. A good fitted component curve is not sufficient to
choose between those physical histories.
