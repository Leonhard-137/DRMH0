# Figure definitions and caveats

## 01_grid_regularization

NK22 Fig.5 analogue. Primary local 4203 points; Nr=50; each map independently divided by its 99th percentile of |deltaT/T|. Full local time span retained. Low-xi amplitudes can exceed the small-perturbation regime; colors alone are not significance.

## 02_moderate_regularization

NK22 Fig.7 analogue, Nt=250. Independent scale per panel. No trajectories are fitted or overlaid: NK22's highlighted velocities were chosen by eye, and are not measurements reproduced here.

## 03_lightcurve_fits

NK22 Fig.6 analogue. Black data, red xi=1, cyan xi=1000. Each band uses the same affine transform for data and both fits: (flux-midrange)/(half-range). No post-fit multiplicative refitting.

## 04_residuals_all_bands

Residual=data-prediction, in observed flux units. Error bars show triplet-adjusted input errors, not full covariance of a fitted residual. Different vertical scales per band. U/u residuals are not automatically BLR light curves.

## 05_coverage_and_kernels

Black HST; red Swift; blue ground. The kernel curves are each normalized to one and color-ordered from short to long wavelength. Broadband coverage does not imply 50 independently resolved radii; neighboring radial kernels overlap strongly.

## 06_paper_numeric_comparison

Fixed physical parameters and regularization, not tuned toward publication. NK22 numbers read from Fig.5 annotations. Input Swift release, wavelengths, time selection and weighting are not identical; deviations do not by themselves diagnose a solver error.

## 07_Uu_exclusion

Identical 50x250 grid and xi=10; U/u measurements removed and remaining bands refit. All three panels use one baseline color scale, including the difference. Agreement tests sensitivity to two bands, not BLR absence in the remaining bands.

## 08_U_residual_lag

2024 Fig.16 analogue. JAVELIN covariance, flat priors with explicit bounds, emcee sampling. Dashed blue lines mark published medians. This conditional likelihood ignores temperature-map uncertainty and shared U/residual noise; it cannot establish that BLR contamination is absent. DRW sigma uses the saved observed flux unit.
