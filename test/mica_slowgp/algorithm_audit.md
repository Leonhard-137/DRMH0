# Independent mathematical audit of the two-band slow-GP core

Date: 2026-10-22. Scope: the isolated Python core in
`/home/leonhard/software/MICA2-slowgp/src/mica_slowgp/core.py` and a read-only
inspection of `make_small_cases.py`. This audit does not establish native-MICA
prior equivalence, sampler convergence, or recovery coverage.

The auditor changed only `test_core.py` and this file. No production file,
existing science input, fitted posterior, or manuscript was changed.

## Model and covariance

The tested model is

\[
y_W(t)=c_W+x(t)+s_W(t),\qquad
y_b(t)=c_b+A_0(G_0*x)(t)+A_1(G_1*x)(t)+s_b(t).
\]

Here `sigma_x` is the stationary OU standard deviation, Gaussian kernels have
unit area on the entire real line, and `A0/A1` are response areas. For
\(\Delta=t_a-t_b\), the covariance of response components is

\[
C_{ab}(\Delta)=\sigma_x^2 A_a A_b\,
J\!\left(\Delta-\mu_a+\mu_b,
\sqrt{w_a^2+w_b^2},\tau_x\right),
\]

where \(J(d,w,\tau)=E_{Z\sim N(0,w^2)}e^{-|d-Z|/\tau}\). The unfiltered
reference uses \((A,\mu,w)=(1,0,0)\). Positive response center therefore
correlates `x(t)` most strongly with a response at `t+mu`, not `t-mu`.
The total response covariance includes both ordered cross terms between
components 0 and 1; their common driver does not make them independent.

Two independent numerical oracles are used in the tests:

1. Direct normal-expectation quadrature, split at the OU absolute-value cusp.
2. Fourier integration of the OU spectrum times Gaussian filtering factors.

Neither oracle evaluates the production error-function closed form. The
zero-width OU identity is used only for unfiltered-process covariance.

The slow kernel is exactly

\[
k_s(d)=(1+|d|/T_s)e^{-|d|/T_s}.
\]

Thus the usual Matérn-3/2 length scale is \(\ell=\sqrt3 T_s\).
`qW/qb` are standard deviations in their respective flux units. Independent
mode has zero cross-band slow prior covariance. Shared mode has
\(s_W=q_Wz,\ s_b=q_bz\), giving cross-covariance \(q_Wq_bk_s\).
The shared reconstruction must preserve this same-process relation in its
joint posterior; drawing these two slow curves independently would be wrong.

## Integrating unknown constants

Let \(K\) include measurement variance and let \(L\) contain the two band
indicator columns. Define

\[
Q=L^TK^{-1}L,\qquad
\widehat c=Q^{-1}L^TK^{-1}y,\qquad r=y-L\widehat c.
\]

For a flat prior on both constants, the tested log likelihood is

\[
\log p(y\mid\theta)=-\frac12\left[
r^TK^{-1}r+\log|K|+\log|Q|+(n-2)\log(2\pi)\right].
\]

The arbitrary common normalization of the improper flat prior is omitted.
In particular, profiling the constants while dropping `log|Q|` does not give
this integrated likelihood. Tests compare its value with dense independent
algebra, verify independent flux-offset invariance and time-origin invariance,
and verify the `n-2` covariance-scale normalization.

For a predicted latent vector with prior covariance \(C_*\), cross-covariance
\(B=\mathrm{Cov}(f_*,y)\), and requested mean design \(L_*\), define
\(D=L_*-BK^{-1}L\). Then

\[
E[f_*+L_*c\mid y]=L_*\widehat c+BK^{-1}r,
\]
\[
\mathrm{Cov}(f_*+L_*c\mid y)
=C_*-BK^{-1}B^T+DQ^{-1}D^T,
\]
\[
\mathrm{Cov}(f_*+L_*c,c\mid y)=DQ^{-1}.
\]

Latent components have zero `L_*`; total W/b curves optionally have their own
constant column. The tests also compare with a broad proper Gaussian prior
on the constants and verify exact joint reconstruction identities, including
the cross-covariance with the constants. Component uncertainty cannot be
added in quadrature as if the reconstructed components were independent.

## Executed checks

`test_core.py` contains 17 tests. All passed in both environments:

| Interpreter | Result | Unittest time |
| --- | --- | --- |
| `/home/leonhard/miniforge3/envs/mica2/bin/python -B` | 17/17 passed | 0.134 s |
| `/home/leonhard/miniforge3/envs/mica2-slowgp/bin/python -B` | 17/17 passed | 0.137 s |

The new-environment run was repeated after the core author's subnormal-number
stability changes; the table records that latest run. An earlier run in that
environment also passed (0.128 s).

The suite covers Gaussian/OU quadrature, zero-width limits, symmetry and tail
finiteness, delay sign, correlated response cross terms, all three observation
covariance modes, positive-semidefinite joint component covariance, exact
zero-slow nesting, both mean normalizers, mean/time translation invariance,
joint conditional means and covariances, mean/component cross covariance,
shared-slow proportionality, and the broad-proper-mean limit.

Run command:

```bash
/home/leonhard/miniforge3/envs/mica2-slowgp/bin/python -B \
  /home/leonhard/sci001/test/mica_slowgp/test_core.py
```

The source defaults to `/home/leonhard/software/MICA2-slowgp/src`; set
`MICA_SLOWGP_SRC` to test another isolated source tree. The tests require only
NumPy, SciPy and the Python standard library. They use tiny synthetic arrays,
fixed inputs and numerical tolerances, create no artifacts, and run no fits.
Evidence class: numerical evidence for the tested implementations and
parameter points, not a universal proof or sampling benchmark.

## Read-only review of the small simulation generator

`make_small_cases.py` uses a stationary exact discrete OU transition. Its
Matérn state is `(s, ds/dt)`, with stationary covariance
`diag(1, 1/Ts**2)` and drift `[[0,1],[-1/Ts**2,-2/Ts]]`; this gives precisely
the slow covariance above. `Q=P-F P F.T` is its exact step-noise covariance.

Gaussian smoothing followed by interpolation at `grid-mu` has the required
positive-delay sign. The grid spans -90 to 180 days, while observations span
approximately 0.35 to 89.65 days. With the stated truth widths and centers,
the eight-sigma convolution support stays well inside the grid for all
observation/prediction times, so boundary reflection and endpoint clipping do
not enter those predictions. The four cases reuse the same OU draw, response
curves, epochs and measurement noise. Shared mode reuses the W-band slow draw
for the response; independent mode uses the separate response slow draw.

The OU/Matérn states are exact on the fine grid, but off-grid linear
interpolation and Gaussian convolution at `dt=0.025 d` are numerical
approximations. This is not an exact continuous-time joint Gaussian sample at
the final observation times. No extra simulation or file write was performed
by this read-only review.

## Scientific and inferential boundaries

- `x` is a latent clean reference-band variation with coefficient fixed to one;
  it is not automatically the physical central illuminating source. Its
  variance is inferred, so there is no additional free reference amplitude.
- Ordered centers remove label symmetry but do not prove that the shorter
  component is disk or the longer component is BLR. Equal/vanishing components
  remain weakly identified, and Gaussian negative-lag tails are retained.
- Shared slow variability can overlap the broad common-driver response in a
  finite light curve. A correct covariance and successful optimizer do not
  establish identifiable physical separation.
- A zero-mean prior does not force a finite trajectory's sample mean to zero.
  The flat constants leave the physical partition of disk/host mean flux
  undetermined. Variable components alone cannot supply absolute disk flux
  for distance inference.
- A single `condition` call is conditional on one hyperparameter point. Full
  posterior predictions must also marginalize those hyperparameters and keep
  cross-component/time covariance. Exact component-sum relations make some
  joint covariance matrices positive semidefinite rather than strictly
  positive definite.
- These checks do not establish prior correctness, native-MICA empirical-DRW
  prior equivalence, evidence normalization across different data/mean spaces,
  nested-sampling convergence, multiple-seed uncertainty coverage, or
  six-band performance. The current implementation and tests are two-band.

## Independent sampler and CLI review (2026-10-22)

Reviewed files: `/home/leonhard/software/MICA2-slowgp/src/mica_slowgp/sampler.py`
and `/home/leonhard/software/MICA2-slowgp/src/mica_slowgp/cli.py`. This appendix
extends the core-only checks above; it does not establish posterior convergence.

### Proper priors and native callback contract

The declared positive-scale priors have finite log-uniform bounds. Their
wrapped proposals are symmetric in log coordinates: the physical proposal
Jacobian cancels the 1/x prior-density ratio, so returning log Hastings factor
zero is correct. Center parameters have a uniform joint density on the bounded
rectangle conditioned on `mu0 < mu1`. `_centers_from_prior` integrates the
allowed upper-center interval to obtain the lower-center marginal, then draws
the upper center conditionally. This is not equivalent to drawing the lower
center uniformly first. Conditional wrapped center moves preserve this joint
prior, and the occasional independent joint-prior refresh also has zero
Hastings correction. Native CDNest is supplied both custom callbacks, rather
than relying on its generic `Uniform` labels for the physical scales.

There is no empirical single-OU W2 prior. Numerical priors are still scientific
assumptions, especially the slow time-scale and TF-width bounds; arithmetic
correctness cannot establish that these choices identify independent slow
variability in a given light curve. Uniform user overrides should stay inside
the physical domain; the core returns minus infinity for invalid parameters.

Native RNG review used `/home/leonhard/software/CDNest/src/dnest.c`: the GSL
pointer is allocated at lines 974-980, while the first custom `from_prior`
callback runs at line 1159 before particle likelihoods and their tiebreakers.
Resetting the RNG at that callback therefore seeds the trajectories. The
earlier floor-level tiebreaker and native equal-weight posterior resampling
remain time dependent and are not primary posterior products.

### Native posterior weights and singleton diagnostic NaNs

The independent verifier is
`/home/leonhard/sci001/test/mica_slowgp/check_native_weights.py`. It reconstructs
the deterministic prior-mass quadrature from `levels.txt` and
`sample_info.txt`, using an independent lexicographic sort. It compares native
normalized weights, finite logX values, and recorded raw log evidence at
native file-print precision. It flags failures and never replaces native
weights or run files. This is a quadrature/output check, not an evidence
reliability or convergence certification.

The source `/home/leonhard/software/CDNest/src/dnestpostprocess.c:274` divides
by N-1 for a level containing N samples. For N=1 this creates a NaN logX
diagnostic coordinate. The singleton's actual quadrature weight, however,
uses the two level boundaries directly at lines 287-304; it does not use the
NaN node. The reader's known singleton midpoint repair is therefore restricted
to logX. All likelihood/weight columns must already be finite, and unexpected
non-finite coordinates fail validation.

For the two existing seed-812 smoke runs under
`/home/leonhard/sci001/output/mica_slowgp/sampler_smoke/replica1` and `replica2`:

- Both have 250 raw samples; zero-based rows 77 and 148 are precisely the two
  singleton levels and the two NaN native logX coordinates.
- Independent versus native normalized weights differ by at most
  3.109e-9 per sample and 5.351e-8 in L1, consistent with native printed precision.
- Recomputed log evidence is -1.7366251586083332, equal to the recorded native
  result at displayed precision. This agreement does not validate its use
  for model selection.
- Physical sample arrays, native weight/logX/log-likelihood arrays (equal
  NaNs included), and sample likelihood columns match between the two seeded
  runs. Native equal-weight resampling is excluded from this check.

The native comparator at `dnestpostprocess.c:476` returns only 0/1 instead of
the qsort three-way contract. This remains a portability concern in the
unmodified dependency. The independent sort-and-weight comparison found no
resulting error in these local smoke runs. Every completed scientific small
run should pass the same verifier; a mismatch must be surfaced, not silently
repaired or dismissed based on the toy agreement.

### Two-band input and joint reconstruction

`cli.read_data` requires exactly three columns per input file and constructs
`PairData` in the correct time/flux/error order. There is no fixed 42-point
assumption or six-band path. A one-rank guard now runs before output writes,
preventing accidental multi-rank input/reconstruction races.

Reconstruction samples hyperparameters using normalized native posterior
weights. For each selected sample it assembles one joint Gaussian containing
all five latent components and both constants, including the latent-offset
cross covariance. The total W2 and response curves are formed from those same
draws, so component cancellation and mean uncertainty are retained. These are
pointwise intervals from finitely many full joint draws, not independent
component error bars.

Review identified an absolute floor of one in the initial PSD tolerance,
which could hide relatively large negative modes when input flux units are
very small. The CLI owner changed this to
`64 * dimension * machine_epsilon * spectral_scale`, records the maximum
tolerance, and clips only modes within that round-off allowance when taking
the sampling square root. Meaningfully negative modes raise an error. The
core covariance itself is neither jittered nor eigenvalue-clipped.

Remaining entry-point limitations are explicit research-prototype boundaries:
positive draw counts and prediction-grid sizes are expected, and the standard
reconstruction grid assumes overlapping observation spans. Invalid choices
or non-overlapping inputs need deliberate handling before extending this
entry point beyond the planned overlapping two-band tests.
