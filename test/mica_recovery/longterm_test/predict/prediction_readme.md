# Real-disk expected-MICA prediction (2026-09-16)

Status: exploratory calculation completed; no jointly calibrated successful design.
No light curves generated and no MICA sampler, restart, or postprocessing launched.
Existing inputs, run settings, priors, and run artifacts were not modified.

## Target and method

Same 458.519-day Mrk817 W2/U input: 314+324 points; original errors normalized by band means.
Targets are Gaussian free and Gamma free100_host, not the older restricted Gamma fit.
Areas below are in fractional-flux units. All widths are standard deviations;
Gamma native theta is std/sqrt(2).

Use the existing Starkey disk_transfer_function separately for W2 (1928 A) and U (3465 A):
inclination 45 degrees, z=.031, T(R)=T1*(R/one light-day)^(-3/4).
T1 is an adjustable design parameter here, not a measured Mrk817 temperature.

For each generator, build the entire joint covariance Ctrue, including noise.
Native MICA C(theta) is evaluated by its compiled Slc/Sll functions, without initialization
of a fit or MPI sampling. With separate marginalized band constants L,
P=C^-1-C^-1 L (L^T C^-1 L)^-1 L^T C^-1,
J=.5[logdet C+logdet(L^T C^-1 L)+tr(P Ctrue)].
An independent Gaussian approximation to the continuum empirical prior is obtained
from the W-only expected-likelihood mode and Hessian. This is not an exact reproduction
of MICA's continuum sampling stage. Two initializations seek compact and boundary-wide
branches; local convergence is not a global-search proof.

## Generator

I0 is unit-variance OU with tauI. Independent s1,s2 are stationary zero-mean Gaussian
processes with C_s(h)=(1+abs(h)/Ts)*exp(-abs(h)/Ts), also independent of I0.
DW and DU are the unit-area real disk kernels. B is unit-area Gamma(k=2,theta=7 d)
shifted by 6 d: its mean is 20 d, standard deviation sqrt(98) d, and all support is causal.

W = alpha DW * [I0 + sqrt(q) s1 + sqrt(q2) s2]
U = alpha [a DU*I0 + b B*I0
            + DU*(hgain sqrt(q) s1(t-delta) + h2 sqrt(q2) s2(t-delta2))]

Thus the BLR response really is driven by I0, while the added shared processes are not.
alpha fixes the expected finite-window, sample-demeaned W intrinsic RMS to .16900624;
measurement errors are not rescaled. q2=0 recovers the one-shared-process family.
Extra U-only smooth noise is available as v,Tu but was not used in the shortlisted cases.

## Results

These predictions are expected-objective modes, not posterior medians.
The observed entries are posterior medians, so equality at prior boundaries is not expected.

| Case | Gaussian broad (area, mean d, std d) | Gamma broad (area, mean d, std d) |
|---|---|---|
| Actual target | .386, 11.26, 29.08 | 7.137, 4.30, 213.72 |
| real_disk_initial | .393, 11.31, 26.13 | 2.417, 0, 216.15 |
| one_shared_refined_temperature | .344, 12.94, 24.09 | 1.724, 0, 216.15 |
| two_shared_gamma_area_candidate | .340, 17.68, 18.33 | 6.763, 0, 216.15 |
| joint_score_iteration1 | .346, 13.97, 23.43 | 3.023, 0, 216.15 |
| joint_score_iteration2 | .270, 19.25, 21.81 | 6.607, 0, 216.15 |

For two_shared_gamma_area_candidate:
T1=4000 K, tauI=30 d, a=.82, b=.15,
q=.8, Ts=70 d, hgain=1.227, delta=3 d,
q2=.045, Ts2=10 d, h2=1.1, delta2=30 d, alpha=.14796233177946055.
Its U finite-window intrinsic RMS is .17179, close to actual .17053.
However the recovered narrow means are .434 d (Gaussian) and .429 d (Gamma),
versus observed median .204 d and .167 d. Gaussian broad negative-delay mass
is only about 16.7%, versus about 34.6% in the target. Therefore it is NOT a successful design.

The first two probes used midpoint disk quadrature (small .1-.3% normalization error).
All later probes use origin-refined cell quadrature. The saved JSON flags this explicitly.
The second score-matching round reached its 35-evaluation limit; subsequent local
likelihood optimizations converged but did not solve the scientific mismatch.

## Verification and uncertainty

For the two-shared candidate, halving the covariance time step from .025 to .0125 d:
maximum covariance entry difference 2.345e-7; mean-projected Gaussian KL 9.383e-6.
Native Gaussian covariance versus independent analytic convolution agrees to 2.08e-17.
Covariance Cholesky succeeded.

Within each generator, compared with the other searched branch:
Gaussian compact expected log-posterior advantage 16.79;
Gamma wide expected log-posterior advantage 42.85.
For these FIXED pairs, random-realization log-likelihood contrast standard deviations
are 58.80 and 73.93 respectively, from Var(delta logL)=.5 tr[((Pbad-Pgood)Ctrue)^2].
These are not MICA recovery probabilities and do not include refitting each realization.

A discarded every-third-observation screening experiment already flipped the baseline
Gamma branch preference. All reported scientific selections and final optimizations
therefore used all timestamps; the sparse screen was not trusted.

Conclusion: the real disk kernel is numerically tractable, but this finite search of
one/two shared stationary-process generators did not jointly reproduce all target
parameters. No theorem of impossibility or physical attribution follows.
Do not promote any candidate to a validated injection configuration on these results alone.

## Files and rerun

These files live in `test/mica_recovery/longterm_test/predict/`.

- predict_disk.py: real-kernel covariance and native-MICA expected-objective projection.
- tune_disk_prediction.py: the second exploratory score-matching round and full-data validation.
- prediction_candidates.json: parameters and numerical fit results for all five main cases.

The prediction code imports `prepare.py` and `starkey_disk_transfer.py` from the
parent experiment directory. Data paths are resolved from the script location;
the candidate JSON stays beside `predict_disk.py`.

From the project root (`/home/leonhard/sci001`):

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/leonhard/miniforge3/envs/mica2/bin/python test/mica_recovery/longterm_test/predict/predict_disk.py --case two_shared_gamma_area_candidate --full-bounds
```

This computes local modes, prints results, and does not run MICA sampling.

To rerun the separate exploratory tuning round from the same project root:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/leonhard/miniforge3/envs/mica2/bin/python test/mica_recovery/longterm_test/predict/tune_disk_prediction.py
```

This starts the tuning optimization with the script's existing parameters.
The simulation configuration remains in `../experiment_real_disk_mismatch.yaml`;
simulation generation, input plotting, and MICA fitting keep their existing commands.
