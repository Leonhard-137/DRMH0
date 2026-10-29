# Observed Mrk817_a original PSD

Six observed bands: UVW2, UVM2, UVW1, U, B, V. Inputs are the unfiltered
test/mica_smooth/data/original/Mrk817_a_*.txt light curves, copied from
Mrk817_a/fflux/host_sub. These had extinction and host subtraction upstream;
"original" here means before Gaussian detrending, not uncalibrated photometry.
The full available interval of each saved band is used, with its own epochs.

Fractional light curve: (F - original mean)/original mean. Unweighted,
floating-mean Lomb--Scargle, normalization=psd, multiplied by 2T/N.
No interpolation, temporal smoothing/detrending, extra mean rescaling, or
noise subtraction is applied. The density normalization is approximate for
irregular sampling; these are observed-window periodograms, not intrinsic PSDs.

Grid: 1/T to 0.30/day, spacing 1/(5T). The ceiling is a display choice, not a
strict irregular-sampling Nyquist frequency. Faint lines are raw ordinates;
black lines are arithmetic mean power in 23 logarithmic frequency bins.
The companion fP plot bins f*P before averaging in the same bins.

Gray dashed curves are the exact expected floating-mean LS periodogram for
independent measurement errors with the original quoted per-epoch variances,
using the actual sample times and the same fractional normalization.
They are measurement-noise expectations, not detection thresholds, confidence
intervals or corrections for red-noise leakage. No PSD slope, break or physical
component is inferred. Fractional normalization differs from the absolute
flux-squared units used in the preceding shared-S simulation figure.

References: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
