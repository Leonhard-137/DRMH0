# Mrk 817 exploratory periodograms

Original input: ../data/original; filtered input: ../cases_highpass/all_s20/prepared. All times are observer-frame days.

Unweighted, floating-mean Lomb–Scargle on fractional flux using the ORIGINAL mean for both series. No interpolation. Astropy psd normalization multiplied by 2T/N: a uniform-sampling-equivalent one-sided fractional PSD normalization, approximate for these irregular times. This is a sampled/dirty periodogram, not a window-deconvolved intrinsic PSD.

Grid: 1/T to 0.30 per day, spacing 1/(5T). Display ceiling is not a strict Nyquist limit. Thin curves are raw ordinates; thick curves are arithmetic means in 23 logarithmic frequency bins. Oversampled ordinates and bins need not be independent; no confidence intervals or significance are claimed.

Noise curves average 64 independent Gaussian measurement-noise realizations using original quoted errors and observation times. Filtered noise is transformed by the same normalized Gaussian matrix (I-W); the restored constant mean is absorbed by LS floating mean. These curves estimate measurement noise only, not red-noise/window uncertainty. Powers are not noise-subtracted.

No PSD slope, break, periodicity or physical component is fitted. Frequency below 1/T is not displayed; longest timescales are poorly sampled. LS power is not additive between trend and residual because of their cross term. The summary low-frequency ratio is a descriptive ratio of sampled periodogram powers below 0.01/day, not an intrinsic variance fraction.

Reference: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
