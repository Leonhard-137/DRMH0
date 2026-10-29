# Randomized QMC proposal-component stratification prototype

This prototype samples the frozen regularized proposal. It changes numerical
integration only; the model, original prior, center strata, Student parameters,
component probabilities and full mixture density remain unchanged. No GP
likelihood or scientific fit was run, and this candidate is not an official
stability gate.

## Sampling and weights

Within each center stratum s, every proposal component k, including the 0.05
defensive prior component, receives a fixed power-of-two count per scramble:

    Nk = 2**max(3, ceil(log2(2048*alpha_k)))

There are R=8 independent randomizations. Each Student component uses a fresh
Sobol engine with dimension d+1, scramble=True, bits=52 and random_base2(log2
Nk). The first d coordinates map through the standard-normal inverse CDF; the
last maps through the chi-square inverse CDF with the unchanged df. Applying
the stored Cholesky factor and chi-square radial scale gives its Student-t
component. The defensive component uses d Sobol coordinates and their logit.
No first point is dropped and no sequence is skipped or thinned.

The seed hierarchy is NumPy SeedSequence(root_entropy,
spawn_key=(stratum_id,scramble_id,component_id)), then a Generator passed to
SciPy Sobol. All child identities and counts are returned and can be saved.
Sobol coordinates equal to 0 or 1 fail explicitly, without clipping or retry.
Subsequent expit saturation in target coordinates is handled by the existing
explicit allow_closed_unit continuous-extension policy; log pi and log q
continue to use the original finite z.

Each row's conditional integral contribution is

    logweight = logL + logpi_z - logq_full_z + logalpha_k - logNk - logR.

The full q denominator includes every Student component and the defensive
prior, irrespective of which component generated the row. Center combination
adds log rho_s. Every component and center stratum is retained. These rows
are quadrature contributions, not iid draws from q; a row-weight ESS is only
descriptive and is not used to estimate uncertainty.

## Error estimates

Each scramble first produces an evidence and observable-numerator integral
vector, with all component contributions summed. Across the R independent
vectors, sample covariance/R estimates covariance of the mean integral.
Posterior means use the ratio of the mean numerator to mean denominator; their
covariance is calculated by the ratio delta method. Center strata use distinct
randomization streams and are combined within each replicate before this
calculation. The full covariance, replicate vectors, scale and column order
are retained, including center posterior mass ratios.

An eight-scramble standard error is an empirical approximation. Ratios have
finite-sample bias, and rare missed contributions can make precision appear
better than it is. A scrambled Sobol calculation does not automatically make
a difficult scientific posterior stable.

## Interfaces and artifacts

`importance.py` is copied byte-for-byte from the regularized candidate.
`qmc_importance.py` adds:

- `component_budget(proposal, target_N=2048, minimum_power=3)`.
- `sample_components(proposal, *, seed, stratum_id=0, target_N=2048,
  scrambles=8)`; returns z, IDs, component probabilities/counts, each row's
  log quadrature factor and the full seed hierarchy.
- `evaluate_u(callback, proposal, *, seed, stratum_id=0, target_N=2048,
  scrambles=8, allow_closed_unit=False, observable_functions=None)`.
- `summarize_scrambles(logweight, scramble_id, scrambles, observables=None,
  saturated=None)`; returns diagnostics and normalized row weights.
- `combine_strata(evaluations, prior_masses)`; requires common R/root seed,
  unique center-stratum IDs, and identical observable names.

The prototype checks are run with:

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      /home/leonhard/miniforge3/envs/mica2-slowgp/bin/python -B \
      /home/leonhard/sci001/test/mica_slowgp/staged_importance_qmc/check_toys.py

The test refuses existing artifacts. The final report is
`artifacts/toy_report.json`; `artifacts/bimodal_primary_rows.npz` retains all
rows from the first bimodal toy. Fixed settings and exact source identities
are recorded in `settings.json`.

## Toy evidence and actual cost

Twenty checks pass, covering inverse maps, endpoint errors, unequal component
counts, exact constant integrand, constant unit-prior likelihood, seed replay,
full-density comparison, all prior strata, and the logweight identity. An
initial exact-constant test exposed sequential bincount summation rounding;
it was replaced by NumPy pairwise sums. The initial report is retained under
`artifacts_initial_bincount`; the target, points and weight formulas did not
change. The corrected exact checks pass at absolute tolerance 1e-14.

For a two-component analytic bimodal likelihood, twelve independent experiments
each used eight scrambles. Evidence truth is 0.9999376818 and the experiment
mean is 0.9999500049. Tail-probability truth is 0.3498977275 and the experiment
mean is 0.3499353426. Between-experiment variance divided by mean reported
variance is 1.48 for evidence and 1.57 for the tail ratio. Twelve of twelve
evidence estimates and eleven of twelve tail estimates are within three
reported standard errors. These are limited calibration observations, not a
new scientific acceptance criterion.

One successful toy suite makes 332,800 cheap analytic likelihood calls and
takes 2.68 seconds. Including the preserved first summation test, the stage
made 665,600 analytic calls. Scientific GP calls remain zero. The 2048 target
is not the actual per-scramble total: power-of-two rounding and the eight-point
floor make the three regularized center strata require 67,520 total rows over
eight scrambles. Exact component counts appear in the report and settings.
For a future budget comparison, compare actual rows; doubling target_N need
not double the realized budget when the floor applies.

This construction is ready for independent numerical review. Any scientific
pilot requires a separate new evaluation, and its outcome cannot be inferred
from these toy checks.
