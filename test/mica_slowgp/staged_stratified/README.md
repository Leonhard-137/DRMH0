# Sorted-center stratified integration prototype

This isolated package is copied from
`/home/leonhard/software/MICA2-slowgp/archive/stability-v2-rslice/src/mica_slowgp`.
Only the copied `nested.py` and `cli.py` are extended, and `stratified.py` is
new. The live package and existing frozen experiments are not modified.
The model remains W2 plus **one** response light curve, with its existing
Gaussian transfer functions, GP covariance, parameter set and likelihood.

## Mathematical contract

The supported original priors have identical uniform center bounds `[a,b]`
and exchangeable `(A0,mu0,w0)` / `(A1,mu1,w1)` priors. All other priors remain
unchanged. The normalized sorted-center prior is

\[
\pi(\mu_0,\mu_1)=\frac{2}{(b-a)^2}
\mathbf 1\{a<\mu_0<\mu_1<b\}.
\]

Exactly three intervals partition the **complete** center range. For one
interval `[l,h]`, its original-prior probability and normalized conditional
transform are

\[
q_j=\frac{(h-a)^2-(l-a)^2}{(b-a)^2},\qquad
\mu_1=a+\sqrt{(l-a)^2+u\big[(h-a)^2-(l-a)^2\big]},
\qquad\mu_0=a+v(\mu_1-a).
\]

Here `u` and `v` are independent unit-uniform coordinates. The Jacobian is
`[(h-a)^2-(l-a)^2]/2`, so this is exactly the normalized conditional prior.
The centers are already ordered; no extra factorial, prior factor or
Jacobian is added to the likelihood. Boundary intersections have zero prior
mass. For `[-3,5,20,28]`, the probabilities are `[64,465,432]/961`.

Let `Zj` be a child's evidence under its normalized conditional prior. Then

\[
Z=\sum_j q_j Z_j,\qquad P_j=\frac{q_j Z_j}{Z}.
\]

Each child's native `logwt` receives exactly one `log(qj)` term before global
`logsumexp` normalization. Equivalently, its normalized posterior weights
are multiplied by `Pj`. No branch is removed, equally reweighted or truncated.
The three conditional runs are not passed to `dynesty.utils.merge_runs`.

With independent conditional log-evidence errors `sj`, the reported
first-order integration error is

\[
\sigma^2_{\log Z}\simeq\sum_jP_j^2s_j^2,\quad
\operatorname{Cov}(\mathbf P)\simeq
J\operatorname{diag}(s_j^2)J^T,\quad
J=\operatorname{diag}(\mathbf P)-\mathbf P\mathbf P^T.
\]

These errors do not include missed modes, inadequate within-stratum mixing,
or conditional sampling bias. Independent seed and budget checks remain
necessary. The strata are numerical regions, not identified physical modes.
Direct conditional ordering also changes unit-cube geometry relative to
sorting unconstrained component triples, while preserving the physical prior.

## Interfaces and budget

- `PriorTransform(..., center_interval=None)` and
  `run_nested(..., center_interval=None)` preserve the existing ordinary path.
  A specified interval enables one normalized conditional run.
- `run_stratified(callback, output_dir, mode, priors, budget, seed,
  center_strata=(-3,5,20,28), resume=False, input_files=None, run_input=None)`
  runs all three children sequentially and combines their full quadratures.
- The copied CLI adds `--center-strata -3 5 20 28` to `fit`; it requires
  `--sampler dynesty`. The original driver/response inputs are stored at the
  parent for the existing gate and reconstruction interface.
- `budget.nlive` is the live-point budget **per stratum**. For example,
  400 means three independent 400-live-point children. The parent records the
  total initial live points separately.
- Child seeds are `SeedSequence(parent_seed).spawn(3)`, with each child's
  `generate_state(1, dtype=uint32)[0]` recorded explicitly.

Use `PYTHONPATH=<this directory>/src` with the isolated Python environment to
select this prototype. No scientific fit was launched during implementation.

## Saved schema

The parent `sampler_config.json` records `integration_scheme`, all original
priors, exact edges and prior masses, parent/child seeds, budget, callback
fingerprint and optional input metadata. `parameterization` includes both
the scheme and edges so the existing stability comparator distinguishes
different integration configurations without changing its thresholds.

The parent weighted CSV has physical parameter columns plus
`log_likelihood`, `conditional_log_prior_mass`, `stratum_id`,
`original_prior_logweight` and globally normalized `weight`.
The raw artifact is `stratified_results.npz`. It records
`conditional_logvol`, conditional and original-prior log-weights, aligned
stratum ids, and the evidence/mass combination. **There is no global
`logvol` trajectory.** `load_saved_posterior` recognizes this schema and
returns its conditional mass explicitly, while reconstruction consumes the
same physical samples and weights as before.

Diagnostics retain every stratum's prior/posterior mass, conditional
evidence, effective sampler settings, row-weight ESS and call counts.
Global weight ESS describes weight concentration, not autocorrelation ESS.
True callback counts and dynesty's reported counts are separate; true counts
sum checkpoint-retained child trajectories and exclude discarded work after
the last saved checkpoint. Reused children add zero current-invocation calls.

## Resume and file ownership

Parent resume requires the complete original configuration and callback
fingerprint, including exact child seeds, interval edges and interval units.
Original input bytes and supplied metadata must also agree. It checks all
present child configurations and completed outputs before continuing work.

- Completed children are validated and read only; samples are not refitted.
- A child with matching configuration and a checkpoint, but no final output,
  resumes its own checkpoint.
- A planned child whose directory is absent is started fresh.
- Partial final output, missing checkpoint/configuration, unknown files or
  inconsistent numerical columns are rejected without overwriting them.
- `completed.json` is written only after all parent output files. A fully
  completed parent can be re-read with `--resume` without rewriting it.
- Existing scheduler-owned `fit.log` and `scheduler_state.json` are allowed
  and are never modified by the integration module. Other pre-existing
  scientific output is not accepted as a fresh run.

## Validation artifacts

`check_stratified.py` runs only analytic likelihoods in temporary folders.
Its durable result is `check_report.json`. Checks include conditional-prior
quadrature, unchanged ordinary transforms, continuous versus interrupted
child-checkpoint recovery, immutable reuse of completed children/parent,
configuration rejection, existing gate-reader schema, and actual CLI routing
with a named analytic replacement callback. The test does not evaluate the
GP likelihood and does not claim a scientific stability-gate pass.

The polynomial toy uses normalized centers `0<x<y<1` and
`L=1+x+2y+3xy`, with `Z=41/12`, posterior means `84/205` and `151/205`.
The recorded transform-integral errors are below `3e-8`. The completed and
resumed paths produce identical physical samples, likelihoods and conditional
volumes, with weights agreeing at `1e-12` relative tolerance.

An independent reviewer additionally used `z=((mu1+3)/31)^2` and `logL=2z`,
whose global evidence is `expm1(2)/2`, and independently checked weight
alignment, invalid configurations and completed-parent reuse. Those review
artifacts are under
`/home/leonhard/sci001/output/mica_slowgp/sampler_smoke/stratified_independent_review/`.
