# Injected and recovered component light curves

All 25 completed simulations from `../../manifest.json` are included. Original M2/B fits live under f9_m2_b; W1/U/V under f9_all_bands. Each PDF page is one response band; rows are five injection conditions; columns compare short/disk, long/slow, and total/noiseless target. PNGs contain the same pages.

## Exact comparison definition

For every existing joint row in `posterior_sample1d.txt_2`, exchange the entire amplitude/centroid/log-width triplet if necessary so component 0 has the shorter centroid. Keep all rows and all other parameters. The centroid column is already the Gamma centroid; do not add twice the width. Initialize the native covariance routines using the same sequence as `src/decompose_mica_branches.py::decompose_conditional_posterior` in a new isolated directory. The helper `reconstruct_component_means.py` uses native data and component cross-covariance matrices and solves for conditional GP means, skipping prediction variances. For each joint draw, q=(L^T C^-1 L)^-1 L^T C^-1 y, alpha=C^-1(y-Lq), and mean_k=U_k alpha + [A_k/sum(A)]q_line. Multiply back by the native target flux scale. No sampling or posterior regeneration occurs. Each reconstruction checks agreement with the existing original native total at its exact exported time grid, tolerance 1e-5 input-flux units. Additional full native sorted-component reconstruction validates response_within/U on 600 time points.

Native light curves include constant baselines, allocated to components by amplitude fraction. Remove each reconstructed curve's arithmetic mean at the exact target observation times, all included in the reconstruction grid. Remove each truth curve's exact mean at those same observation times. Keep amplitudes and times unchanged. The plots thus compare variable flux, not absolute component DC levels. The known injected noiseless target is 1 + disk_variation + slow_variation. Truth files and input flux/time columns are checked against each other. Time is observer-frame days; units are the original synthetic normalized flux units, mean approximately 1. These experiments did not inject an interband physical SED or a special U excess.

## Reading limits

- Native means condition on the observed driver AND target. A recovered long curve following an independent injected slow curve does not establish physical response or causality.
- The injected driver is simulated using F9 sampling/noise properties. These are recovery experiments, not decompositions of the real F9_a measured flux.
- In independent-slow cases there is a true slow light curve but no true response transfer function for that component.
- Solid curves are posterior averages of conditional GP means, not a single median-parameter prediction. No credible interval is drawn. Native component error columns are max(across-draw SD of conditional means, maximum conditional GP SD); they are neither quantiles nor complete uncertainty of the centered curve. The mean-only helper does not compute or export them.
- response_within/W1 and response_within/U have Kish ESS approximately 6.9 and 11.6 (12 and 19 exported posterior rows). Their plots are tentative and marked LOW ESS. Completion of sampling alone did not guarantee stable posteriors.
- The reconstruction includes every actual target observation time and a grid of 1 day or finer; truth is displayed on its saved 0.2-day grid. Observation gaps and noise limit interpretation of fine-scale details; do not attribute all discrepancies to transfer-function failure.
- `comparison_metrics.csv` is descriptive only: RMSE and correlation compare centered means with exact truth at actual target observation times. They are not calibrated tests and do not use uncertainty weights. A zero slow truth has no normalized RMSE or correlation.

## Posterior provenance

| Case | Band | Exported posterior rows | Kish ESS from raw weights | Rows relabelled |
| --- | --- | ---: | ---: | ---: |
| disk_only | M2 | 922 | 702.9 | 292 |
| disk_only | W1 | 971 | 679.7 | 429 |
| disk_only | U | 1166 | 834.2 | 365 |
| disk_only | B | 1015 | 761.1 | 505 |
| disk_only | V | 1036 | 656.4 | 648 |
| response_within | M2 | 2824 | 2549.8 | 2477 |
| response_within | W1 | 12 | 6.9 | 12 |
| response_within | U | 19 | 11.6 | 15 |
| response_within | B | 2170 | 1692.6 | 310 |
| response_within | V | 788 | 524.8 | 144 |
| random_within | M2 | 929 | 524.2 | 307 |
| random_within | W1 | 1128 | 810.4 | 558 |
| random_within | U | 1396 | 1152.5 | 554 |
| random_within | B | 1369 | 759.6 | 828 |
| random_within | V | 1165 | 809.3 | 386 |
| response_beyond | M2 | 2008 | 1382.2 | 883 |
| response_beyond | W1 | 1459 | 996.3 | 659 |
| response_beyond | U | 844 | 601.0 | 575 |
| response_beyond | B | 1292 | 1023.5 | 1033 |
| response_beyond | V | 1648 | 1138.8 | 153 |
| random_beyond | M2 | 1220 | 775.7 | 871 |
| random_beyond | W1 | 1229 | 1004.4 | 454 |
| random_beyond | U | 947 | 650.2 | 358 |
| random_beyond | B | 1070 | 741.8 | 369 |
| random_beyond | V | 958 | 578.7 | 761 |

Reproduction: `/home/leonhard/miniforge3/envs/astro/bin/python -B test/mica_recovery/f9_all_bands/plot_component_lightcurves.py --jobs 4`. Completed isolated reconstructions are reused. `--plot-only` regenerates plots and CSV from them. Native stdout is retained per reconstruction directory. Original fit artifacts are never overwritten.
