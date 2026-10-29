# Fixed covariance-regularized importance proposal

This candidate applies one predetermined covariance shrinkage to the retained
`staged_importance_refined` 12+12 component proposal. It does not rerun EM or
evaluate a GP likelihood. Original proposals, source files and pilot outputs
were checked unchanged after processing. The model, physical priors, three
conditional strata and scientific target are unchanged.

The source refinement trained on half the pooled three-seed NS posterior and
half the first IS pilot posterior, with separate posterior and posterior times
negative-lag-fraction blocks. The fitted components exposed very low effective
training sample sizes: in the middle stratum, six tail components had
responsibility ESS below 12. The unregularized candidate is retained; it was
not independently evaluated as a scientific pilot.

## Exact operation

For each block, independently, let pi_k be its normalized component weights,
m_k its means, S_k its stored Student-t shape and e_k its recorded training
responsibility ESS. With nu=7, a=1.2 and d=11, first recover the training
Gaussian covariance:

    C_k = S_k / [a^2 (nu-2)/nu]
    m_block = sum_k pi_k m_k
    C_block = sum_k pi_k [C_k + (m_k-m_block)(m_k-m_block)^T]
    alpha_k = (d+2)/(e_k+d+2)
    C_new,k = (1-alpha_k) C_k + alpha_k C_block
    S_new,k = a^2 (nu-2)/nu C_new,k

The between-mean term is calculated in the original logit coordinates, after
removing Student conversion from the within covariances. It is not multiplied
by the inverse scale conversion. The value d+2 is a fixed proposal-regularizing
heuristic, not a new scientific prior or a fit to the known simulation truth.

Every component retains its mean and full probability bit for bit. Full block
probabilities remain 0.475 base, 0.475 tail and 0.05 defensive logistic prior.
Each Student density is normalized for positive-definite shape. Their positive
full probabilities sum to one, so the full proposal is normalized analytically.
No importance weight is clipped, and no original posterior row is reevaluated
as a fresh scientific draw.

## Saved implementation and interface

`importance.py` is byte-identical to the preceding refined implementation.
The frozen proposal format and `FrozenProposal.load`, `sample_z`, `logpdf_z`
interfaces are unchanged. Only `regularize.py` is new. The entry point is:

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      /home/leonhard/miniforge3/envs/mica2-slowgp/bin/python -B \
      /home/leonhard/sci001/test/mica_slowgp/staged_importance_regularized/regularize.py

It refuses an existing output directory. Frozen proposals are
`artifacts/regularized/stratum_00.npz` through `stratum_02.npz`. The matching
`stratum_XX_covariance_audit.npz` contains unscaled training covariance,
regularized training covariance, block mean, within/between/total block
covariance, component ESS, alpha and the Student conversion factor. The
machine-readable report is `artifacts/regularized/regularization_report.json`.

## Numerical checks and density placement

Postprocessing and proposal-only checks took 0.188 seconds, with zero EM and
zero GP calls. All 72 component shapes are positive definite. The total full
probability error is at most 1.11e-16; means and full weights match bit for bit.
Save/load density values match exactly. An independent SciPy Student density
calculation differs by at most 8.44e-14 in log q on the selected rows. For each
stratum, 1024 fixed-seed proposal draws and their full densities are finite.
These checks verify numerical construction, not Monte Carlo convergence.

Components with training responsibility ESS below 12 carry 2.641%, 15.084%
and 6.482% of the full proposals in strata 00, 01 and 02. These probabilities
remain unchanged by shrinkage; they describe the fitted components' training
support. The threshold 12 is descriptive, not an acceptance gate.

For middle-stratum pilot row 5507, log q relative to the first proposal rises
by 7.975 after regularization, compared with 39.639 before regularization.
The corresponding changes for rows 128 and 7952 are 9.832 and 3.677 (formerly
39.780 and 4.930). All nine selected row comparisons are retained in the report.
These are in-sample density placement measurements. They establish neither
reduced variance nor a reliable future ESS.

Independent variance review of the first pilot found that row 5507 dominated
the tail-mean variance, while other high-weight rows with nearly zero tail
fraction affected the denominator. An old NS source already contained a near
neighbor of row 5507; it was not evidence of an entirely new mode. These facts
motivated keeping both base and tail blocks. No additional local tuning was
performed after the fixed shrinkage rule was selected.

## Evidence boundary

Only a new independent evaluation under this frozen proposal can test whether
the shrinkage reduces scientific importance variance. Poorly supported
components have been broadened, which can lower density elsewhere and increase
the number of low-likelihood draws. Defensive prior support preserves the
target in the importance identity but does not guarantee practical efficiency
or that a finite pilot will reveal all rare weights. This stage does not claim
posterior stability, physical-component recovery or a Hubble-constant result.
