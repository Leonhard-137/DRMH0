# Ten-source temperature mapping: figure captions

## Scope and interpretation

These are regularized inverse-model temperature maps, not direct images of the disk. No propagation trajectories, wave speeds, significance contours or pixel uncertainties are inferred by this plotting script. Color saturation is display-only; fit arrays are not clipped. No pixel-level posterior uncertainties are supplied.

The baseline is Nr=50, Nt=250 and xi=10 unless a source summary explicitly records otherwise. A common xi does not mean equivalent physical smoothing across sources or time-grid sizes: cadence, baseline length, errors, kernels and priors differ. The shared time axis notation is observer days since each source's own first observation, not a synchronized event or rest-frame timeline.

Low/high-pass maps use the supplied filtered data fits. Smoothing induces correlated noise, and that covariance has not been propagated into a new likelihood. Their nominal chi-squared values must not be interpreted as independent-noise goodness-of-fit tests or compared as formal evidences. Finite coverage, gaps and endpoint extension are especially relevant when 28/56 days approaches the campaign duration.

Independent q99 normalization is only used to compare morphology. It cannot support a claim of equal amplitude, equal significance, absence of disk variability or identification of disk/BLR/magnetosonic-wave components. Parameters, input provenance and quality warnings remain in the input manifest, physical parameter file and per-source summaries.

## Baseline scales and coverage

Absolute-map saturation is the fraction of all radial-time cells with |deltaT/T| > 0.2; it is not a fraction of statistically significant or independently resolved pixels.

| Source | Bands | Data | Span (d) | Nt=250 spacing (d) | q99 | chi2/N | Absolute saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mrk817_a | 17 | 6135 | 420.101 | 1.6872 | 0.16258 | 1.1139 | 0.304% |
| Mrk142 | 11 | 2308 | 230.820 | 0.9270 | 0.080619 | 0.7709 | 0.000% |
| Mrk335 | 10 | 2457 | 185.295 | 0.7442 | 0.2007 | 1.1693 | 1.008% |
| F9_a | 6 | 1863 | 708.588 | 2.8457 | 0.12743 | 0.5999 | 0.000% |
| Mrk279_b | 6 | 1034 | 172.380 | 0.6923 | 0.1196 | 0.4143 | 0.000% |
| Mrk509 | 6 | 1409 | 272.642 | 1.0949 | 0.14534 | 0.6173 | 0.000% |
| NGC5548 | 19 | 4203 | 288.974 | 1.1605 | 0.17405 | 1.3696 | 0.616% |
| NGC4151_a | 6 | 1693 | 69.254 | 0.2781 | 0.062015 | 0.4063 | 0.000% |
| NGC6814 | 6 | 1511 | 74.760 | 0.3002 | 0.088569 | 0.3471 | 0.000% |
| NGC2617_a | 6 | 257 | 49.650 | 0.1994 | 0.10629 | 0.3748 | 0.000% |

## Source-specific warnings

### Mrk817_a

- This is Mrk817_a (about420 d), not the full524-day reference. The original full817 reproduction is unchanged.
- New errors use NK22 mean floors, whereas the full817 reference used NK24 median floors. Compare with awareness of both changes.
- Time bounds use matched published HJD, NOT an assumed MJD+0.5 conversion.
- The Mrk817_a source label/selected segment is not by itself evidence that its input rows equal the full 17-band paper campaign; data selection is determined separately.
- At least one band has a gap >=28 days; Gaussian filtering interpolates across gaps.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

### Mrk142

- Lijiang lV (62 rows) excluded: arbitrary flux normalized to mean 1, NOT physical f_lambda (Cackett2020 Table2).
- Ground r/i/z counts differ from paper Table1; this is the supplied local screened release, not an exact published selection.
- Swift wavelength convention follows project config (M2=2246,U=3465 A); paper Table3 gives 2236,3467 A.
- Ground monitoring extends beyond Swift; those epochs have weaker radial information.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Super-Eddington accretion invalidates a strict radiatively efficient thin-disk interpretation; treat the recovered map as a conditional formal benchmark.
- Local standard-lag benchmark M=1.95e6 Msun and L_bol/L_Edd=3.40 uses a different literature compilation; it is intentionally not substituted for the NK22 mapping parameters.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

### Mrk335

- Swift physical-flux units adopt the existing project convention; Swift calibration was not independently reconstructed in this run. Ground-table physical-flux units were explicitly verified.
- Ground griz are the supplied local intercalibrated release; counts and some flux/error values differ from Kara2023 published Table2.
- Time origin is retained as local MJD. Cross-instrument UTC/TDB and heliocentric-conversion provenance is not fully documented. Published ground-table comparisons show telescope-dependent offsets; sub-day propagation interpretation needs a separate timing audit.
- z combines z-prime/z_s; 8700 A is the paper Table3 effective-wavelength approximation, not a measured transmission kernel.
- Ground monitoring extends beyond Swift; those epochs have weaker radial information.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Ground_i retains 2 same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.
- Ground_z retains 1 same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.
- Mass and Eddington ratio follow the project's documented compilation, not the distinct parameters used for the theoretical lag example in Kara et al. 2023.
- The luminosity distance is cosmology-assumed, with no peculiar-velocity or flow correction.
- 99th percentile \|deltaT/T\| >0.2: inspect the small-perturbation approximation; 0.2 is a descriptive flag, not a validated rejection threshold.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

### F9_a

- Late Fairall9 campaign only; local screened/cropped release differs from Wang2025 segment2 and is NOT the early Fairall9 dataset mapped in NK22.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Local F9_a is a later Swift segment (MJD 58550.52-59259.11), not NK22's earlier Swift plus optical campaign: this is method transfer.
- The local late-campaign benchmark uses M=2.60e8 Msun and lambda_Edd=0.035; NK22's older normalization is deliberately retained and is not asserted to be a simultaneous luminosity estimate.
- 50 radial cells over [1,400] do not exactly reproduce NK22's trimmed radial discretization.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

### Mrk279_b

- Segment2 only; no data from Mrk279_a or segment3 are merged.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Historical mass/accretion inputs are transferred to the local Swift segment 2, not refitted to its luminosity state.
- The luminosity distance is cosmology-assumed, with no peculiar-velocity or flow correction.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

### Mrk509

- 2017 Swift campaign; six UVOT bands only in this local source table.
- Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Local standard-lag benchmark uses M=1.1e8 Msun and lambda_Edd=0.095 from a later study; NK22's 0.05 is deliberately retained for mapping.
- 50 radial cells over [1,400] do not exactly reproduce NK22's trimmed radial discretization.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

### NGC5548

- Later public5900-point TempMap bundle is NOT used: it mixes Swift MJD and HST/ground HJD and includes another Swift release.
- This input is4203 original observed points; no extinction or host correction.
- Public wavelength vector is inherited; NK22 text lists slightly different U/u/z wavelengths. This is already tested in the individual5548 reproduction.
- The local Swift light curves are the 2019 reprocessing, not automatically identical to the input version used by NK22.
- The separate local standard-lag benchmark uses logM=7.72 and lambda_Edd=0.05; the existing NK22 mapping values are deliberately preserved.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

### NGC4151_a

- 2016 Swift campaign only; no later NGC4151_c data are merged.
- Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- The local later standard-lag benchmark adopts M=2.30e7 Msun and lambda_Edd=0.014; those are not substituted for NK22's mapping inputs.
- 50 radial cells over [1,400] do not exactly reproduce NK22's trimmed radial discretization.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

### NGC6814

- No additional dereddening is applied by this pipeline despite substantial Galactic reddening; upstream correction history is not independently established, and temperature amplitudes are conditional on the supplied-flux convention.
- Gonzalez2024 correction (stae1353) changes UV calibration and withdraws the extreme blue-spectrum/truncation conclusion of the original paper.
- Zenodo14930797 has erroneous NGC6814 bands. Local counts and first time/flux/error rows of ALL six bands match corrected record20430106, Lightcurves_new/NGC6814(All); entire-array equality was not checked. No re-calibration or detrending applied.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- This transfers the mass, distance and fiducial luminosity normalization, not Gonzalez et al.'s relativistic irradiated-disk model: their i=45 deg, spin and color correction are not implemented.
- Published lag/SED behavior disfavors a self-consistent standard-disk explanation; map structures remain conditional on the deliberately simplified kernels.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

### NGC2617_a

- 2013 outburst Swift subset only (about50 d); no later NGC2617 campaign is merged.
- Assumed project f_lambda units; original Vega zero-point/band conversion and whether prior dereddening occurred have NOT been reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- The Eddington ratio is an approximate epoch scaling with fixed bolometric correction, not a new fit to the 2013 SED.
- A changing-look outburst need not satisfy the steady thin-disk baseline assumption.
- The luminosity distance is cosmology-assumed, with no peculiar-velocity or flow correction.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

## Atlas page index and captions

### Page 1: 01_baseline_absolute

Baseline temperature maps in the exact manuscript sample order. Each panel uses its own full observed time span; horizontal widths do not represent equal durations. A single absolute color scale, deltaT/T = [-0.2, 0.2], applies to all sources. Values outside this range saturate, not disappear; saturated cell fractions are listed below. 

[PDF](figures/01_baseline_absolute.pdf) · [300-dpi PNG](figures/01_baseline_absolute.png) · [110-dpi preview](figures/01_baseline_absolute_preview.jpg)

### Page 2: 02_baseline_normalized

Baseline temperature maps in the exact manuscript sample order. Each panel uses its own full observed time span; horizontal widths do not represent equal durations. Each source is divided by its own baseline 99th percentile of |deltaT/T|. The common displayed range [-1,1] is dimensionless and does not imply equal physical amplitudes. 

[PDF](figures/02_baseline_normalized.pdf) · [300-dpi PNG](figures/02_baseline_normalized.png) · [110-dpi preview](figures/02_baseline_normalized_preview.jpg)

### Page 3: 03_regularization_1

Four regularization strengths; all cells in a source row use that source's xi=10 baseline q99, not per-panel renormalization. All requested panels are available.

[PDF](figures/03_regularization_1.pdf) · [300-dpi PNG](figures/03_regularization_1.png) · [110-dpi preview](figures/03_regularization_1_preview.jpg)

### Page 4: 03_regularization_2

Four regularization strengths; all cells in a source row use that source's xi=10 baseline q99, not per-panel renormalization. All requested panels are available.

[PDF](figures/03_regularization_2.pdf) · [300-dpi PNG](figures/03_regularization_2.png) · [110-dpi preview](figures/03_regularization_2_preview.jpg)

### Page 5: 04_filter28_1

Original, 28-day-FWHM low-pass and corresponding high-pass reconstructions. A source row shares its original xi=10 q99 scale. All requested panels are available.

[PDF](figures/04_filter28_1.pdf) · [300-dpi PNG](figures/04_filter28_1.png) · [110-dpi preview](figures/04_filter28_1_preview.jpg)

### Page 6: 04_filter28_2

Original, 28-day-FWHM low-pass and corresponding high-pass reconstructions. A source row shares its original xi=10 q99 scale. All requested panels are available.

[PDF](figures/04_filter28_2.pdf) · [300-dpi PNG](figures/04_filter28_2.png) · [110-dpi preview](figures/04_filter28_2_preview.jpg)

### Page 7: 05_time_grid_1

Nt=100/250/500 maps, displayed on their actual native time grids and the common source-baseline q99 scale. A fixed xi across grid sizes is not a fixed continuum regularization strength; this is sensitivity, not a convergence demonstration. All requested panels are available.

[PDF](figures/05_time_grid_1.pdf) · [300-dpi PNG](figures/05_time_grid_1.png) · [110-dpi preview](figures/05_time_grid_1_preview.jpg)

### Page 8: 05_time_grid_2

Nt=100/250/500 maps, displayed on their actual native time grids and the common source-baseline q99 scale. A fixed xi across grid sizes is not a fixed continuum regularization strength; this is sensitivity, not a convergence demonstration. All requested panels are available.

[PDF](figures/05_time_grid_2.pdf) · [300-dpi PNG](figures/05_time_grid_2.png) · [110-dpi preview](figures/05_time_grid_2_preview.jpg)

### Page 9: 06_band_subsets_1

Available band-subset sensitivity maps at the baseline color scale. Blank optional panels denote an unavailable test and do not denote zero signal. Parameter and subset definitions are inherited from the fit summary, not inferred from morphology. All requested panels are available. Primary Swift-only repeats: F9_a, Mrk279_b. For these sources the primary data already are the same six Swift bands: the Swift-six map is a copy of the baseline, not a separate refit or experiment. A map correlation of one is tautological and is not evidence of robustness.

[PDF](figures/06_band_subsets_1.pdf) · [300-dpi PNG](figures/06_band_subsets_1.png) · [110-dpi preview](figures/06_band_subsets_1_preview.jpg)

### Page 10: 06_band_subsets_2

Available band-subset sensitivity maps at the baseline color scale. Blank optional panels denote an unavailable test and do not denote zero signal. Parameter and subset definitions are inherited from the fit summary, not inferred from morphology. All requested panels are available. Primary Swift-only repeats: Mrk509, NGC4151_a, NGC6814, NGC2617_a. For these sources the primary data already are the same six Swift bands: the Swift-six map is a copy of the baseline, not a separate refit or experiment. A map correlation of one is tautological and is not evidence of robustness.

[PDF](figures/06_band_subsets_2.pdf) · [300-dpi PNG](figures/06_band_subsets_2.png) · [110-dpi preview](figures/06_band_subsets_2_preview.jpg)

### Page 11: Mrk817_a/01_low_high_pass

Mrk 817 (a): Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.162582. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/Mrk817_a/01_low_high_pass.pdf) · [300-dpi PNG](figures/Mrk817_a/01_low_high_pass.png) · [110-dpi preview](figures/Mrk817_a/01_low_high_pass_preview.jpg)

### Page 12: Mrk817_a/02_lightcurve_fits

Mrk 817 (a): every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/Mrk817_a/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/Mrk817_a/02_lightcurve_fits.png) · [110-dpi preview](figures/Mrk817_a/02_lightcurve_fits_preview.jpg)

### Page 13: Mrk817_a/03_residuals

Mrk 817 (a): baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/Mrk817_a/03_residuals.pdf) · [300-dpi PNG](figures/Mrk817_a/03_residuals.png) · [110-dpi preview](figures/Mrk817_a/03_residuals_preview.jpg)

### Page 14: Mrk817_a/04_coverage_kernels

Mrk 817 (a): actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/Mrk817_a/04_coverage_kernels.pdf) · [300-dpi PNG](figures/Mrk817_a/04_coverage_kernels.png) · [110-dpi preview](figures/Mrk817_a/04_coverage_kernels_preview.jpg)

### Page 15: Mrk142/01_low_high_pass

Mrk 142: Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.0806193. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/Mrk142/01_low_high_pass.pdf) · [300-dpi PNG](figures/Mrk142/01_low_high_pass.png) · [110-dpi preview](figures/Mrk142/01_low_high_pass_preview.jpg)

### Page 16: Mrk142/02_lightcurve_fits

Mrk 142: every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/Mrk142/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/Mrk142/02_lightcurve_fits.png) · [110-dpi preview](figures/Mrk142/02_lightcurve_fits_preview.jpg)

### Page 17: Mrk142/03_residuals

Mrk 142: baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/Mrk142/03_residuals.pdf) · [300-dpi PNG](figures/Mrk142/03_residuals.png) · [110-dpi preview](figures/Mrk142/03_residuals_preview.jpg)

### Page 18: Mrk142/04_coverage_kernels

Mrk 142: actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/Mrk142/04_coverage_kernels.pdf) · [300-dpi PNG](figures/Mrk142/04_coverage_kernels.png) · [110-dpi preview](figures/Mrk142/04_coverage_kernels_preview.jpg)

### Page 19: Mrk335/01_low_high_pass

Mrk 335: Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.200697. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/Mrk335/01_low_high_pass.pdf) · [300-dpi PNG](figures/Mrk335/01_low_high_pass.png) · [110-dpi preview](figures/Mrk335/01_low_high_pass_preview.jpg)

### Page 20: Mrk335/02_lightcurve_fits

Mrk 335: every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/Mrk335/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/Mrk335/02_lightcurve_fits.png) · [110-dpi preview](figures/Mrk335/02_lightcurve_fits_preview.jpg)

### Page 21: Mrk335/03_residuals

Mrk 335: baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/Mrk335/03_residuals.pdf) · [300-dpi PNG](figures/Mrk335/03_residuals.png) · [110-dpi preview](figures/Mrk335/03_residuals_preview.jpg)

### Page 22: Mrk335/04_coverage_kernels

Mrk 335: actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/Mrk335/04_coverage_kernels.pdf) · [300-dpi PNG](figures/Mrk335/04_coverage_kernels.png) · [110-dpi preview](figures/Mrk335/04_coverage_kernels_preview.jpg)

### Page 23: F9_a/01_low_high_pass

Fairall 9 (a): Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.127432. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/F9_a/01_low_high_pass.pdf) · [300-dpi PNG](figures/F9_a/01_low_high_pass.png) · [110-dpi preview](figures/F9_a/01_low_high_pass_preview.jpg)

### Page 24: F9_a/02_lightcurve_fits

Fairall 9 (a): every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/F9_a/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/F9_a/02_lightcurve_fits.png) · [110-dpi preview](figures/F9_a/02_lightcurve_fits_preview.jpg)

### Page 25: F9_a/03_residuals

Fairall 9 (a): baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/F9_a/03_residuals.pdf) · [300-dpi PNG](figures/F9_a/03_residuals.png) · [110-dpi preview](figures/F9_a/03_residuals_preview.jpg)

### Page 26: F9_a/04_coverage_kernels

Fairall 9 (a): actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/F9_a/04_coverage_kernels.pdf) · [300-dpi PNG](figures/F9_a/04_coverage_kernels.png) · [110-dpi preview](figures/F9_a/04_coverage_kernels_preview.jpg)

### Page 27: Mrk279_b/01_low_high_pass

Mrk 279 (b): Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.119602. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/Mrk279_b/01_low_high_pass.pdf) · [300-dpi PNG](figures/Mrk279_b/01_low_high_pass.png) · [110-dpi preview](figures/Mrk279_b/01_low_high_pass_preview.jpg)

### Page 28: Mrk279_b/02_lightcurve_fits

Mrk 279 (b): every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/Mrk279_b/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/Mrk279_b/02_lightcurve_fits.png) · [110-dpi preview](figures/Mrk279_b/02_lightcurve_fits_preview.jpg)

### Page 29: Mrk279_b/03_residuals

Mrk 279 (b): baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/Mrk279_b/03_residuals.pdf) · [300-dpi PNG](figures/Mrk279_b/03_residuals.png) · [110-dpi preview](figures/Mrk279_b/03_residuals_preview.jpg)

### Page 30: Mrk279_b/04_coverage_kernels

Mrk 279 (b): actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/Mrk279_b/04_coverage_kernels.pdf) · [300-dpi PNG](figures/Mrk279_b/04_coverage_kernels.png) · [110-dpi preview](figures/Mrk279_b/04_coverage_kernels_preview.jpg)

### Page 31: Mrk509/01_low_high_pass

Mrk 509: Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.145337. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/Mrk509/01_low_high_pass.pdf) · [300-dpi PNG](figures/Mrk509/01_low_high_pass.png) · [110-dpi preview](figures/Mrk509/01_low_high_pass_preview.jpg)

### Page 32: Mrk509/02_lightcurve_fits

Mrk 509: every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/Mrk509/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/Mrk509/02_lightcurve_fits.png) · [110-dpi preview](figures/Mrk509/02_lightcurve_fits_preview.jpg)

### Page 33: Mrk509/03_residuals

Mrk 509: baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/Mrk509/03_residuals.pdf) · [300-dpi PNG](figures/Mrk509/03_residuals.png) · [110-dpi preview](figures/Mrk509/03_residuals_preview.jpg)

### Page 34: Mrk509/04_coverage_kernels

Mrk 509: actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/Mrk509/04_coverage_kernels.pdf) · [300-dpi PNG](figures/Mrk509/04_coverage_kernels.png) · [110-dpi preview](figures/Mrk509/04_coverage_kernels_preview.jpg)

### Page 35: NGC5548/01_low_high_pass

NGC 5548: Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.174051. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/NGC5548/01_low_high_pass.pdf) · [300-dpi PNG](figures/NGC5548/01_low_high_pass.png) · [110-dpi preview](figures/NGC5548/01_low_high_pass_preview.jpg)

### Page 36: NGC5548/02_lightcurve_fits

NGC 5548: every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/NGC5548/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/NGC5548/02_lightcurve_fits.png) · [110-dpi preview](figures/NGC5548/02_lightcurve_fits_preview.jpg)

### Page 37: NGC5548/03_residuals

NGC 5548: baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/NGC5548/03_residuals.pdf) · [300-dpi PNG](figures/NGC5548/03_residuals.png) · [110-dpi preview](figures/NGC5548/03_residuals_preview.jpg)

### Page 38: NGC5548/04_coverage_kernels

NGC 5548: actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/NGC5548/04_coverage_kernels.pdf) · [300-dpi PNG](figures/NGC5548/04_coverage_kernels.png) · [110-dpi preview](figures/NGC5548/04_coverage_kernels_preview.jpg)

### Page 39: NGC4151_a/01_low_high_pass

NGC 4151 (a): Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.0620146. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/NGC4151_a/01_low_high_pass.pdf) · [300-dpi PNG](figures/NGC4151_a/01_low_high_pass.png) · [110-dpi preview](figures/NGC4151_a/01_low_high_pass_preview.jpg)

### Page 40: NGC4151_a/02_lightcurve_fits

NGC 4151 (a): every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/NGC4151_a/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/NGC4151_a/02_lightcurve_fits.png) · [110-dpi preview](figures/NGC4151_a/02_lightcurve_fits_preview.jpg)

### Page 41: NGC4151_a/03_residuals

NGC 4151 (a): baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/NGC4151_a/03_residuals.pdf) · [300-dpi PNG](figures/NGC4151_a/03_residuals.png) · [110-dpi preview](figures/NGC4151_a/03_residuals_preview.jpg)

### Page 42: NGC4151_a/04_coverage_kernels

NGC 4151 (a): actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/NGC4151_a/04_coverage_kernels.pdf) · [300-dpi PNG](figures/NGC4151_a/04_coverage_kernels.png) · [110-dpi preview](figures/NGC4151_a/04_coverage_kernels_preview.jpg)

### Page 43: NGC6814/01_low_high_pass

NGC 6814: Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.0885691. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/NGC6814/01_low_high_pass.pdf) · [300-dpi PNG](figures/NGC6814/01_low_high_pass.png) · [110-dpi preview](figures/NGC6814/01_low_high_pass_preview.jpg)

### Page 44: NGC6814/02_lightcurve_fits

NGC 6814: every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/NGC6814/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/NGC6814/02_lightcurve_fits.png) · [110-dpi preview](figures/NGC6814/02_lightcurve_fits_preview.jpg)

### Page 45: NGC6814/03_residuals

NGC 6814: baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/NGC6814/03_residuals.pdf) · [300-dpi PNG](figures/NGC6814/03_residuals.png) · [110-dpi preview](figures/NGC6814/03_residuals_preview.jpg)

### Page 46: NGC6814/04_coverage_kernels

NGC 6814: actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/NGC6814/04_coverage_kernels.pdf) · [300-dpi PNG](figures/NGC6814/04_coverage_kernels.png) · [110-dpi preview](figures/NGC6814/04_coverage_kernels_preview.jpg)

### Page 47: NGC2617_a/01_low_high_pass

NGC 2617 (a): Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. All six panels use the original xi=10 q99=0.106288. Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.

[PDF](figures/NGC2617_a/01_low_high_pass.pdf) · [300-dpi PNG](figures/NGC2617_a/01_low_high_pass.png) · [110-dpi preview](figures/NGC2617_a/01_low_high_pass_preview.jpg)

### Page 48: NGC2617_a/02_lightcurve_fits

NGC 2617 (a): every fitted band, black observed points with gray capless input-error bars; red xi=1 and cyan xi=1000 model predictions at the actual observation times. Lines connect these predictions only; no continuous between-observation prediction was recomputed. One affine transform per band is applied identically to data, errors and both models: subtract the data midrange and divide by max(data half-range, median input error). No multiplicative refit, outlier clipping or fixed vertical clipping is applied.

[PDF](figures/NGC2617_a/02_lightcurve_fits.pdf) · [300-dpi PNG](figures/NGC2617_a/02_lightcurve_fits.png) · [110-dpi preview](figures/NGC2617_a/02_lightcurve_fits_preview.jpg)

### Page 49: NGC2617_a/03_residuals

NGC 2617 (a): baseline residuals divided by the input per-point errors used in the fit. The gray error bars have unit length in these units; they are not the full posterior residual covariance. Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.

[PDF](figures/NGC2617_a/03_residuals.pdf) · [300-dpi PNG](figures/NGC2617_a/03_residuals.png) · [110-dpi preview](figures/NGC2617_a/03_residuals_preview.jpg)

### Page 50: NGC2617_a/04_coverage_kernels

NGC 2617 (a): actual per-band observation times (no inferred continuous coverage) and the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, not physical component identity. Kernel overlap means that the radial grid cells are not independently resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions where simultaneous multiband coverage is absent.

[PDF](figures/NGC2617_a/04_coverage_kernels.pdf) · [300-dpi PNG](figures/NGC2617_a/04_coverage_kernels.png) · [110-dpi preview](figures/NGC2617_a/04_coverage_kernels_preview.jpg)

## Export validation

50 atlas pages; every page exported as vector PDF, 300-dpi PNG and 110-dpi JPEG. Render time: 200.8 seconds. See figures/render_validation.json for raster dimensions, stored DPI and text-bound checks. These automated bounds checks do not replace visual inspection.
