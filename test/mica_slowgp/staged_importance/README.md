# Frozen defensive importance proposal candidate

This directory is an isolated candidate implementation. It imports no GP or
scientific sampler and changes no existing model, prior, source or fit output.
Only analytic evaluation and proposal training have been run here.

## Proposal and estimator

For each normalized conditional prior, `u` is its unit-cube coordinate and
`z=logit(u)`. The transformed prior is

`log p_z(z) = sum[-logaddexp(0,-z_i)-logaddexp(0,z_i)]`.

Each trained Gaussian mixture is converted to a multivariate Student-t mixture
with `df=7`. If the fitted Gaussian covariance is C, the stored Student-t shape
matrix is `S = 1.2**2 * C * (df-2)/df`, so its covariance is `1.2**2*C`.
Base and tail blocks each have six components. Their **full** probabilities
are 0.475 and 0.475; the defensive logistic-prior component has probability
0.05. `component_weights` sum to 0.95. There is no additional multiplication
by 0.95 in the density or sampling code.

After training, all distribution arrays are frozen. Independent evaluation
draws use the full mixture density in

`log w = log L(T(u)) + log p_z(z) - log q_z(z)`.

No weight clipping, smoothing, training-row reuse or zero-weight removal is
performed. The conditional evidence estimator is `mean(w)`. For strata with
different evaluation sizes, **each global row** has unnormalized mass
`p_stratum * w / N_stratum`. `combine_strata` applies this denominator and keeps
all three strata. Independent conditional evidence variances are added after
the squared prior-mass factors. Posterior means are self-normalized ratio
estimates, hence they are not exactly unbiased at finite N. Empirical iid
standard errors cannot reveal important weights that were never sampled.

## Interfaces

The standalone module is `importance.py`; add this directory to the import
path without installing a package.

```python
from importance import FrozenProposal, evaluate_u, combine_strata

proposal = FrozenProposal.load("artifacts/trained/stratum_00.npz")
z = proposal.sample_z(n, rng)  # np.random.Generator; shape (n, d)
logq = proposal.logpdf_z(z)

evaluation = evaluate_u(
    log_likelihood_u, proposal, n, seed,
    allow_closed_unit=True,
    observable_functions={"tail": tail_fraction_u},
)
# Each target/observable callback accepts one u row and returns one scalar.
# Only an explicitly supported continuous closed-unit target should use True.
combined = combine_strata(evaluations, [64/961, 465/961, 432/961])
```

`evaluate_u` returns all rows of `z`, `u`, `logu`, `log1mu`, `logp_z`, `logq_z`,
`logl`, `logw`, normalized `weights`, `component_id`, `saturated`, observable
arrays, and diagnostics. Defensive-prior rows have component ID K, where K is
the number of Student-t components. `sample_z(...,return_components=True)`
returns `(z,component_ids)`.

`fit_gaussian_mixture(z,weights,n_components=6,seed=...,max_iter=60)` uses
weighted EM in standardized coordinates. Its small covariance ridge changes
the proposal only. `freeze_student_mixture(fits,block_weights,epsilon=.05,
df=7,inflation=1.2)` requires the **full** block probabilities to sum to
`1-epsilon`. Frozen proposals save/load as NPZ without pickle. Metadata record
the sources, weights, random seeds, covariance convention and coordinate map.

## Extreme unit coordinates

Log prior and proposal densities always use the original finite z, with stable
log-sigmoid expressions. z is never clipped, and proposal samples are never
redrawn because of expit saturation. Ordinary finite-precision PRNG draws are
used; the defensive prior uses 52-bit open midpoint uniforms.

The default evaluation policy is strict: if float64 expit returns 0 or 1, it
raises before calling the target. With explicit `allow_closed_unit=True`, the
natural expit result is passed to a target that supports a continuous extension
on the closed cube. Saturated rows remain present, and their count,
unnormalized evidence contribution and normalized importance mass are reported.
They are never automatically assigned zero likelihood. This policy avoids
silently truncating or altering q; it still evaluates a finite-precision target
and is not a claim of arbitrary-precision equivalence near endpoints.

Computational arrays retain mathematical `-inf` for zero likelihood/weight.
For standards-compliant JSON reports, `json_safe` converts nonfinite report
values to null. A null logarithm of a zero contribution means `log(0)=-inf`;
it does not change any estimator array.

## Completed checks and training

`check_toys.py` completed 17 checks in about 8.35 seconds. These cover comparison
with SciPy's normalized multivariate-t density; one-dimensional integration of
the full defensive mixture; sampled covariance conversion; constant and bimodal
known targets; a tail observable; frozen save/load; unequal per-stratum N and
zero weights; and explicit `z=+/-1000` handling. Results are in
`artifacts/toy_checks.json`. Training and evaluation RNG seeds are distinct.

`train_existing.py` read the completed 400-live child results for parent seeds
71231, 71247 and 71263. Within each conditional layer, each seed's posterior
weights were normalized independently and the three posteriors were mixed
equally. The prior inverse is the existing sorted conditional transform;
amplitudes and widths are not permuted. Base weights are `w_post`; tail weights
are `w_post*f_minus`, where

`f_minus=[A0*Phi(-mu0/w0)+A1*Phi(-mu1/w1)]/(A0+A1)`.

Only exact zero-weight training rows may be removed; positive-weight rows on
unrepresentable unit boundaries cause an explicit error rather than clipping.
No such error occurred. Original input NPZ SHA256 values are stored in each
proposal and `artifacts/trained/training_report.json`.

Training took about 16.56 seconds and made zero GP calls. Frozen proposals are
`artifacts/trained/stratum_00.npz`, `stratum_01.npz`, and `stratum_02.npz`.
Their base weighted training ESS values are approximately 9507, 11161 and
12821; tail ESS values are approximately 5039, 302 and 486. These are training
descriptors, not importance-sampling ESS or evidence precision. The scripts
refuse to overwrite frozen NPZ files; replay should use a fresh directory copy
containing the Python source files.

No scientific pilot has been started by these scripts. Candidate training,
correct density formulas and passing analytic tests do not establish scientific
convergence, tail coverage or useful evaluation efficiency.
