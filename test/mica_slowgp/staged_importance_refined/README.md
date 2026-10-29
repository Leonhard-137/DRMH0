# Refined frozen importance proposal candidate

This is a new candidate directory. The frozen first proposal, first scientific
pilot, scientific source code, model and priors remain unchanged. No GP call or
new scientific evaluation is performed by `train_refined.py`.

## Motivation and fixed choices

The first pilot (`importance_pilot_v1/seed_82103_n16384`) estimated the negative
lag fraction mean as 0.0016972664 with empirical MCSE 0.0001297221. Its global
Kish ESS was 1137.4. The lowest conditional layer had one normalized row weight
0.045565 and its top 1% rows carried 34.69% of that layer's weight. These facts
show concentration; they do not locate all of the tail estimator's variance.

The independent root/variance-agent review subsequently attributed 95.488% of
the tail-mean variance to the middle layer, and 4.46% to the lowest. Middle-layer
row 5507 alone contributed approximately 64.622% of global tail-mean variance:
its f_minus was 0.0295747, global posterior weight about 0.003739, w0=0.100959
near its prior lower bound, mu1=8.5074, w1=5.5976, A0=0.70583, A1=0.60156.
Other high-weight rows with f_minus approximately zero affect the denominator.
This supports retaining both a base block and a tail block, rather than fitting
only a tail block.

One fixed training candidate is used, with no selection on a new evaluation:

- Original three normalized NS conditional posteriors each have training mass
  1/6, for a combined NS mass 1/2.
- The first IS pilot's normalized weights **within each stratum** have training
  mass 1/2. Global stratum weights are not used inside a conditional fit.
- Base training uses these combined weights. Tail training multiplies them by
  the same analytic f_minus before normalization.
- Each block has 12 Gaussian training components, increased from six. EM has
  max_iter=100, tolerance=1e-5, standardized covariance regularization=1e-4.
- Frozen Student-t conversion retains df=7 and SD inflation=1.2.
- Full proposal block probabilities remain 0.475 base, 0.475 tail and 0.05
  defensive prior. Thus each frozen layer has 24 Student-t components.
- Training seeds are 870100/870101, 870200/870201, 870300/870301 for
  base/tail in layers 00/01/02.

Twelve components provide additional local flexibility without immediately
using sixteen components on a small effective tail training sample. Keeping
half the training mass in the old NS results limits domination by a single IS
pilot. Neither choice guarantees lower future importance variance.

## Implementation and source separation

`importance.py` was copied from the frozen `archive/importance-pilot-v1` module.
Only the Gaussian-training metadata were extended: final objective, tolerance
convergence, whether max_iter was reached without convergence, and final
component responsibility ESS. The sampling, full mixture density, covariance
conversion, importance weighting and stratum-combination functions are
unchanged. `train_refined.py` is new and owns all new training artifacts.

NS rows use the existing conditional sorted-center inverse transform. IS rows
use their original stored z; they are not sent through another inverse transform
or clipped. A/w identities are not permuted. Exact zero-weight training rows
may be excluded from the training objective. No positive training weight is
clipped or capped.

Source NPZ hashes and the original pilot configuration/diagnostics hashes are
checked before and after training. New frozen outputs are exclusively under
`artifacts/trained/`, with a new `training_report.json`. Save/load density
agreement is checked without evaluating a likelihood.

The report includes descriptive changes in log q on a few original pilot rows,
including the middle-layer row 5507. Those rows were used for training: their
new density values are **not** independent efficiency checks, new importance
weights, new evidence estimates or convergence evidence. The next pilot must
use new independent evaluation draws and its own frozen proposal identity.

## Cost and interpretation

This candidate increases proposal training and density evaluation work, while
performing zero new GP evaluations. Doubling the number of mixture components
does not double the cost of a full scientific draw: it affects q sampling/pdf,
not the likelihood. The actual runtime is reported rather than assumed.

By comparison, with a stable finite iid variance, halving an MCSE by increasing
N alone requires approximately four times as many independent likelihood
evaluations. Refining q is a possible way to improve that cost, but the first
pilot's rare weights make this an experiment, not a promised speedup.

EM convergence means only that its weighted proposal-training objective met
the tolerance. Hitting max_iter, component ESS, covariance regularization and
training data limitations must be reported. None certifies posterior or
evidence convergence. The defensive prior preserves full proposal support even
if the fitted mixture misses a region, while empirical future MCSE still may
fail to detect unsampled rare weights.
