# Fixed-posterior influence of target-only signals in MICA

This diagnostic applies three existing B-band MICA posterior distributions to
the same two known independent injection vectors and twelve deterministic
sin/cos probes. It changes no fit, posterior, observing time, measurement error,
or original input file. The native reconstruction configuration is staged in
new subdirectories here.

The three fit families are `disk_only`, `random_within`, and `random_beyond`,
with 1015, 1369, and 1070 existing equal-weight joint posterior rows. Each row
has already been relabelled by increasing transfer-function centroid. The
corresponding original-weight Kish ESS values are 761.08, 759.60, and 741.78.

## Definition

At fixed parameter draw theta, let C be the native covariance of the observed
W2 and B data, L the two constant-baseline design columns, and U_k the native
cross covariance of component k at the B observation times with all observed
data. Define

    Q_C = C^{-1} - C^{-1} L (L^T C^{-1} L)^{-1} L^T C^{-1}.

The target-only perturbation is `e_u = [0_W2, u_B / B_scale]`. After removing the
arithmetic mean at the actual B observation times, its component influence is

    delta_k = B_scale * D * U_k * Q_C * e_u.

D removes only a constant. Native MICA's allocation of the fitted baseline to
each component is constant here, so it vanishes under D. The actual native
baseline term is included before centering as an additional implementation
check. All covariances and scales stay fixed when u is added or removed. This
does not include the change of the posterior that a new fit would produce.

Reported coefficients and norms are calculated from the posterior-mean delta:

- `beta_k = dot(delta_k, u) / dot(u, u)`: amplitude projection coefficient. It
  is not a power fraction, is not coherence, and need not belong to [0,1].
- `r_k`: Pearson correlation at the actual observation times.
- `rms_over_u_k`: RMS(delta_k) / RMS(u).
- `residual_rms_over_u`: RMS(u - delta_short - delta_long) / RMS(u).
- `beta_*_q16/q50/q84`: quantiles over the existing fixed-posterior sensitivity
  coefficients. These are not uncertainties after refitting modified data.

Both actual independent u vectors have RMS 0.06173568777 in the normalized
input flux units. The sin/cos vectors have periods 2, 5, 10, 30, 100, and 300 d,
are centered at the actual sampling, and scaled to this same RMS. They are
deterministic operator probes, not added random simulations, Fourier estimates,
or evidence that the light curves contain periodic components. Their sin/cos
differences also reflect the finite irregular observing window.

## Main results for the known independent signals

| Fixed posterior | Applied u | beta_short | beta_long | Residual RMS / RMS(u) |
|---|---|---:|---:|---:|
| disk_only | random_within u | 0.462613 | 0.011250 | 0.556600 |
| disk_only | random_beyond u | 0.447447 | 0.024256 | 0.548502 |
| random_within | random_within u | 0.024967 | 0.946157 | 0.069584 |
| random_within | random_beyond u | 0.014212 | 0.967100 | 0.028751 |
| random_beyond | random_within u | 0.303066 | 0.346417 | 0.402514 |
| random_beyond | random_beyond u | 0.027984 | 0.945682 | 0.068293 |

The same independent vector maps differently under the three fitted covariance
models. Independence does not make the short component insensitive to target
variation. Under the random_within posterior, the 10 d sin/cos probes have
short-component beta 0.3105/0.3174, while 100 d probes have short beta
0.0867/0.0765 and long beta 0.8187/0.8358. Thus the behavior also depends on the
temporal pattern of u; the numbers are conditional sensitivities, not estimates
of the unknown independent contribution in a real light curve.

## Difference between the two actual fitted short curves

`paired_short_effects.json` additionally compares the already fitted short curve
of each independent-signal case with the pure-disk fitted short curve at their
318 identical B observation times. Disk and noise realizations match; the
observed-input difference equals the corresponding u to within 9.70e-12. This
paired end-to-end difference includes changes in the fitted parameter/posterior
distribution. Subtracting the fixed-posterior direct delta gives a fitted-model
change remainder; this is not an independently identified physical component.

| Case | Difference vector | beta versus u | RMS / disk RMS | RMS / u RMS |
|---|---|---:|---:|---:|
| random_within | Actual fitted short difference | 0.021633 | 0.031147 | 0.103823 |
| random_within | Fixed-posterior direct delta | 0.024967 | 0.018504 | 0.061681 |
| random_within | Fitted-model change remainder | -0.003334 | 0.028837 | 0.096124 |
| random_beyond | Actual fitted short difference | 0.000419 | 0.020177 | 0.067258 |
| random_beyond | Fixed-posterior direct delta | 0.027984 | 0.020404 | 0.068013 |
| random_beyond | Fitted-model change remainder | -0.027564 | 0.012964 | 0.043212 |

The true disk RMS is 0.20578563; u RMS is 0.06173569. The actual fitted-short
changes have RMS about 3.11% and 2.02% of the injected disk RMS in this one paired
realization. They are differences of two independently sampled fits and include
posterior numerical error. Neither these RMS ratios nor the beta coefficients
are population contamination rates or disk-recovery acceptance tests. In
particular, the fixed-posterior coefficient near 2.5% is not the total effect of
adding u and refitting. The vectors and beta coefficients add; their RMS values
do not, because direct and remainder contributions can reinforce or cancel.

## Verification and limits

For the first parameter draw of every posterior, the native
`decompose_single_component` routine was independently called on the full data
and on data with each true u removed. Only the in-memory normalized data vector
was changed, and it was restored immediately. The difference agrees with the
linear calculation: maximum discrepancy over all these checks is
1.19e-13 input-flux units.

The complete posterior-mean reconstructions of the original data were also
recomputed and compared at every exact observation time against the previously
validated component means. The largest difference is 1.56e-15. Per-fit details
are recorded in `summary.json`. All 3454 posterior rows were retained.

The optional driver-only variance fraction eta was deliberately not calculated.
In the installed MICA source, `set_covar_Amat_line_single` fills off-diagonals
with `Sll_single` (component-versus-total covariance) and its diagonal with
`Sll_single2` (component-versus-itself covariance). Consequently its complete
matrix was not assumed to be K_kk for a centered trace calculation. This does
not enter the U_k Q_C mean calculation above; no native variance code was changed.

## Files

- `all_summary.csv`: 42 diagnostic rows across the three fixed posteriors.
- `all_results.json`: complete definitions, provenance, and validation results.
- `<fit_case>/influence_curves.npz`: exact observation times, all input vectors,
  posterior-mean delta_short/delta_long/delta_total, per-draw beta coefficients,
  and probe names. The curve matrices have shape (318, 14), and beta_draws has
  shape (N_posterior, 2, 14).
- `<fit_case>/summary.csv` and `summary.json`: per-fit results.
- `<fit_case>.log`: execution record; new native setup files are under
  `<fit_case>/native_*/`.

Reproduction entry point: `../../analyze_independent_influence.py`. Its
controller launches three single-rank MPI workers with numerical-library
threads fixed to one. Existing completed outputs are reused by the controller.
