# Local PSD peak check: figure captions

## local_peak_examples_remaining.png / .pdf

Each row compares one saved candidate in W2 (left) and U (right). Black: observed binned PSD. Blue: a deliberately selected single Monte Carlo realization. Red short dashes: the pointwise ensemble median. Pale grey: the first eight ordinary saved draws in their original order. All panels show raw, unfiltered spectra on the same PSD scale.

**Selection disclosure:** The blue curves were intentionally chosen from the simulated ensemble by the upstream local-peak check. They are illustrative matches, not random or necessarily representative realizations. Their visual agreement must not be used to infer a match probability. Candidate frequencies are reported separately in candidate_rates.csv and the rate figure; the exact local-peak rule is recorded in definition.json. Screening uses 500 realizations per candidate. The displayed blue examples, red medians and first-eight grey draws come from a separate 2000-realization confirmation ensemble with a new seed. The example was chosen for low whole-spectrum mismatch among confirmation draws passing the joint local-peak rule. This plotter performs no new selection.

PSD estimator and normalization are inherited unchanged from the supplied arrays: unweighted floating-mean Lomb-Scargle × 2T/N, with arithmetic averaging within 23 logarithmic frequency bins, in fractional-flux units. No extra smoothing or normalization is applied by this plotting script.

### Rows

1. `C075`; role `highest_screen_rate`; tauI=32 d, tauS=5 d; selected zero-based realization index=1283; ensemble size=2000.
2. `C089`; role `highest_screen_rate`; tauI=8 d, tauS=160 d; selected zero-based realization index=1692; ensemble size=2000.
3. `C069`; role `previous_fast_example`; tauI=0.1 d, tauS=20 d; selected zero-based realization index=1901; ensemble size=2000.

## local_peak_rates.png / .pdf

Each point is an evaluated candidate, placed by its central-driver and independent-slow time scales. Color gives the saved joint similar-realization rate as a percentage. Rates and their statistical definition are read from the upstream check, not recomputed from the selected example curves. These are 500-draw screening rates, distinct from the 2000-draw confirmation rates. Coincident parameter locations may overlap. The feature and tolerance were chosen after inspecting the observations: rates are model-conditional replication fractions, not p-values or posterior probabilities.
