# Mrk 817 observed PSD before and after Gaussian detrending

These are real observations, not recovery simulations. The original files in
../data/original are already extinction-corrected and host-subtracted. The two
filtered series are read directly from ../cases_highpass/all_s20/prepared and
../cases_highpass/all_s40/prepared, the actual saved MICA inputs. All six bands,
including UVW2, use F_highpass = F_original - W F_original + mean(F_original).
W is the row-normalized Gaussian matrix at each band's actual epochs. Its sigma
is 20 or 40 observer-frame days. No new filtered inputs or fits are produced.
Times, errors and the high-pass formula are checked against each original file.

Estimator matches ../plot_original_psd.py: unweighted floating-mean Lomb--Scargle,
normalization=psd, multiplied by 2T/N. Fractional flux always uses the ORIGINAL
mean for all three curves within each band; no unit-variance rescaling is used.
Each band's full saved window and own epochs are retained. No interpolation,
additional detrending, noise subtraction or fitted PSD model is used.
Frequency grid: 1/T to 0.30/day, spacing 1/(5T); 0.30/day is a plotting limit,
not an exact irregular-sampling Nyquist frequency. Curves show arithmetic mean
periodogram power in the same 23 logarithmic bins. The fP companion averages
f*P in each bin, rather than multiplying binned P by the bin-center frequency.
All panels within a figure share axis limits. Colors and styles distinguish
processing choices, not different physical components or a preferred model.

Raw and binned periodograms are saved as CSVs. The reference PSD is numerically
checked against the previously plotted original PSD files. The summary's low
frequency ratios are descriptive sums of periodogram power below 0.01/day over
the same sampled grid, not intrinsic variance fractions or significance tests.
These finite-window irregularly sampled periodograms are approximate fractional
spectral densities; leakage and frequency-bin correlations remain. There are
no uncertainty bands or measurement-noise curves in these comparison figures.
No claim of a statistically significant interband PSD difference is made.
