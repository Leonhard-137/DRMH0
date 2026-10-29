# Randomized component-stratified Sobol: independent mathematical review

Decision: suitable for one new independent scientific pilot. No blocking mathematical finding was identified. This review does not establish scientific posterior stability or pass the frozen gate; it does not choose between the existing nested-sampling and importance-sampling estimates.

The target integral is estimated by

`(1/R) sum_r sum_k (alpha_k/N_k) sum_i [L(z_rki) pi_z(z_rki) / q_z(z_rki)]`,

where every component, including the defensive prior, is sampled explicitly and `q_z=sum_k alpha_k q_k` remains the full mixture denominator. Since `sum_k alpha_k E_qk[h/q]=integral h`, unequal power-of-two component allocations do not change the target. The original scientific prior mass `rho_s` is inserted once when center strata are merged.

For each independently scrambled replicate r, the implementation reconstructs the integral vector `(Z_r,Y_1r,...)` before averaging. Its mean covariance is the sample covariance across R vectors divided by R. Posterior-ratio gradients retain numerator/denominator covariance. Independent center streams may be paired by r and summed before this calculation; the resulting empirical covariance is a valid randomization-error estimate. It does not turn rows inside a scramble into iid samples.

The d+1 map uses d independent normal inverse CDFs and one chi-square(df) inverse CDF. Multiplication by the Cholesky factor and division by `sqrt(chi2/df)` gives the multivariate Student t with the stored shape matrix S; its covariance is `df/(df-2)*S`. The defensive component uses the exact logistic transform. Open-uniform endpoints fail visibly without clipping, replacement, or data-dependent reseeding. SeedSequence uses distinct `(stratum,r,component)` spawn keys; same-key reproducibility and changed-stratum streams were checked.

Seven independent synthetic checks passed, with zero scientific GP likelihood calls:

1. A hand-calculated four-vector example reproduces evidence 2.5, evidence variance 5/12, ratio 1.7, and ratio variance 47/750, including cross-observable ratio covariance.
2. Splitting every row into 17 equal contributions leaves evidence and MCSE unchanged.
3. A deterministic Student-t map point and the defensive logistic map match their analytic values.
4. Full-mixture density agrees with independent SciPy Student-t and logistic densities.
5. A synthetic target equal to the full proposal produces exact unit importance ratios and unit evidence, validating alpha/N/R factors and full-q cancellation.
6. Endpoints 0/1 are rejected; seed keys are unique and deterministic without stream reuse across strata.
7. Hand-specified center integrals combined with rho reproduce the global ratio, stratum mass and ratio variance.

The actual frozen proposal budget is:

| Center stratum | Rows per scramble | R | Total rows |
|---|---:|---:|---:|
| 0 | 2696 | 8 | 21568 |
| 1 | 3088 | 8 | 24704 |
| 2 | 2656 | 8 | 21248 |
| Total | 8440 | 8 | 67520 |

These counts exclude any separate precision-audit likelihood calls in the scientific wrapper. `target_N=2048` is a nominal component allocation; minimum-eight and power-of-two rounding increase the actual budget.

R=8 provides only seven covariance degrees of freedom, so estimated errors can remain noisy and the multivariate empirical covariance has rank at most seven. Self-normalized ratio estimates are biased at finite budget and their delta-method errors are approximate. Rare missed contributions can still make apparent precision misleading. Natural expit saturation must follow an explicit recorded closed-unit policy; it is distinct from raw Sobol endpoints, which fail. Any completed pilot must preserve these qualifications and be compared with the existing inconsistent estimates, not promoted to a gate pass.

The JSON records the reviewed source hashes, exact checks and per-component budgets. No core source or proposal was edited by this reviewer.
