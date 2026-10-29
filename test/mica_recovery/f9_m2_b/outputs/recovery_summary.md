# F9 M2/B slow-signal injection recovery

Status: **8/8 fits analyzed**. Generated 2026-09-05T05:48:24.997905+00:00.

Values below are weighted medians [15.865%, 84.135%]. Time quantities are observer-frame days. All joint raw samples retain their posterior weight; no branch selection or prior-based rejection was applied.

| Band | Case | Smaller centroid | Larger centroid | Larger width | Total centroid | Weight ESS |
|---|---|---:|---:|---:|---:|---:|
| M2 | response_within | 0.0913 [0.05677, 0.1184] | 61.57 [57.67, 65.88] | 31.83 [29.44, 34.48] | 33.68 [30.16, 37.52] | 2549.8 |
| M2 | random_within | 0.09989 [0.0742, 0.1326] | 1413 [934.5, 1818] | 20.48 [13.98, 31.73] | 451.1 [288.4, 668.7] | 524.2 |
| M2 | response_beyond | 0.1013 [0.07518, 0.1278] | 585.8 [380.2, 963] | 297 [196.7, 481.2] | 455.8 [231.6, 849.3] | 1382.2 |
| M2 | random_beyond | 0.1088 [0.07623, 0.1416] | 752.4 [109.3, 1553] | 330.6 [193.2, 545.9] | 607.8 [85.59, 1175] | 775.7 |
| B | response_within | 0.961 [0.8727, 1.061] | 56.02 [49.44, 62.93] | 31.87 [27.35, 35.83] | 31 [25.67, 36.47] | 1692.6 |
| B | random_within | 0.9791 [0.8898, 1.095] | 1141 [504.2, 1711] | 17.69 [10.9, 33.47] | 362.2 [218.5, 567.6] | 759.6 |
| B | response_beyond | 1.074 [0.9927, 1.162] | 806.1 [408.4, 1436] | 346.2 [202, 585.5] | 624.4 [269.8, 1104] | 1023.5 |
| B | random_beyond | 1.105 [1.014, 1.184] | 874.1 [291.8, 1641] | 405.4 [196.2, 650.4] | 691.2 [192.4, 1322] | 741.8 |

## Interpretation and limits

- The two components are ordered by centroid within each complete joint draw. Calling the smaller one a recovered disk is an identification hypothesis, not a consequence of this relabeling.
- The injected short-reference centroid is the absolute band-kernel centroid minus the W2-kernel centroid. The finite-difference disk transfer ratio is not assumed to be an exact positive two-Gamma kernel. There is no exact injected Gamma width for the disk component.
- True-response cases have a known slow Gamma component and formal total relative moments. Independent-signal cases have no injected total transfer function or total lag. A fitted slow component there demonstrates possible misattribution in this realization, not a measured false-positive probability.
- Broad configured priors are centroid −10 to 2000 d and width 0.01 to 1200 d for both components; the tables and charts use each run's native exported bounds. Width/amplitude edge fractions are measured in log coordinates, centroid edges in linear coordinates. An outer-5%-strip mass is a screen, not evidence by itself of truncation or a nonphysical fit.
- Importance-weight ESS describes concentration of saved-sample weights. Neither high ESS, multiple particles, nor 10000 saved samples establishes convergence or coverage of all posterior modes. Native particle IDs are not independent chains.
- Native amplitudes describe MICA-normalized curves. Input-flux gain is native amplitude times target normalization divided by driver normalization; only this restored gain is compared with injected area. Total centroid and variance do not change under this common gain factor.
- Total moments are calculated analytically for each full-support Gamma mixture and then summarized. They never come from integrating the standard −100 to 500 d exported plot, which truncates the long-response support.
- Both slow-timescale tiers retain the same F9 cadence and baseline. The distinction concerns the amount of baseline covering the slow variability; it is not a strict Nyquist test for irregular observations.
- This is one paired stochastic realization at one slow/disk RMS. Quantities outside the data-supported frequency range remain model/prior dependent.

## Injected references

| Band | Case | Short relative centroid | Slow centroid | Slow width | Total relative centroid |
|---|---|---:|---:|---:|---:|
| M2 | response_within | 0.127983 | 60 | 30 | 30.6287 |
| M2 | random_within | 0.127983 | not defined | not defined | not defined |
| M2 | response_beyond | 0.127983 | 800 | 400 | 694.649 |
| M2 | random_beyond | 0.127983 | not defined | not defined | not defined |
| B | response_within | 1.17902 | 60 | 30 | 30.1067 |
| B | random_within | 1.17902 | not defined | not defined | not defined |
| B | response_beyond | 1.17902 | 800 | 400 | 689.231 |
| B | random_beyond | 1.17902 | not defined | not defined | not defined |

## Screens and incomplete outputs

- response_within/M2: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- random_within/M2: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- response_beyond/M2: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- random_beyond/M2: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- response_within/B: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- random_within/B: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- response_beyond/B: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.
- random_beyond/B: At least one native parameter has >10% posterior mass in an outer 5% prior strip; inspect prior sensitivity.

## Files

- `recovery_summary.json`: complete provenance, native priors, weighted intervals, truth and diagnostics.
- `recovery_parameters.csv`: component/total intervals and physical prior bounds.
- `recovery_prior_edges.csv`: sorted-component prior-edge mass in the native sampling coordinate.
- `F9_M2_recovery_parameters.png/pdf` and `F9_B_recovery_parameters.png/pdf`: paired comparisons, with injected references and native prior ranges.
