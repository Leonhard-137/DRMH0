# Total light-curve PSD comparison

Shared-S tau2, area9 simulation: W2 = disk + S + noise; U = disk + BLR + S + noise.
Each total periodogram is calculated directly from its saved input light curve;
component periodograms are not summed. The constant baseline is removed by the
floating-mean Lomb--Scargle fit.

This plot reads the existing `*_sampled_psd.csv` total columns. Unweighted
floating-mean Lomb--Scargle with normalization=psd, multiplied by 2T/N; no
interpolation, no separate variance normalization, and no noise subtraction.
These are finite-window spectra, not window-deconvolved intrinsic PSDs.

Top: arithmetic mean P(f) in the original 29 logarithmic frequency bins.
Bottom: arithmetic mean f*P(f) in exactly the same bins. fP indicates power per
log-frequency interval, subject to the sampled-periodogram limitation above.

Curves show raw and both-band Gaussian detrending at sigma=5/10/20/40 days.
The sixth case u_s20 overlaps raw exactly in W2 and all_s20 exactly in U, so no
duplicate curve is drawn. Filter formula: original minus Gaussian trend plus
original mean. This plotting script does not alter inputs or fitting jobs.
