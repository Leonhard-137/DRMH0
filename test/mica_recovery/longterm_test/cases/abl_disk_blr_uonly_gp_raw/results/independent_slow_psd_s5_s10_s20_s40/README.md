# Independent slow-signal PSD

Source: `cases/abl_disk_blr_uonly_gp_raw/components/U_components.txt`, zero-indexed column 2
(`shared_long_variation` is the legacy name; here it is the U-only independent GP).
Only the saved slow-signal contribution is plotted: no disk, BLR, or measurement-noise terms.
All selected sigma curves read the existing paired detrended component files. Direct
reapplication of their normalized Gaussian weights is checked against these files.
No new realization, input edit, MICA sampling, or parameter change was performed.

## Top panel: known ensemble PSD

The actual generator uses covariance C(h)=A^2(1+|h|/tau)exp(-|h|/tau), with
A=0.15396862368414638 and tau=55.3735295468 observer-frame days.
Under Fourier convention exp(-2*pi*i*f*t), the ONE-SIDED PSD for f>0 is
P(f)=8*A^2*tau/[1+(2*pi*tau*f)^2]^2; its integral equals A^2.
This is a smooth spectrum, not a narrow line at period=tau. The knee is defined
by 2*pi*tau*f=1 (one quarter of the zero-frequency plateau).
The ideal filtered spectra multiply this PSD by
[1-exp(-2*pi^2*sigma^2*f^2)]^2. They assume an infinitely long, continuously
sampled stationary signal. They are not expected to exactly reproduce the
finite-window residual realization in the lower panel.

## Bottom panel: the signal actually injected

The 324 saved U epochs are irregular. We use unweighted floating-mean
Astropy LombScargle(normalization='psd'), without interpolation, times 2*T/N.
This is the project's uniform-sampling-equivalent one-sided density normalization;
it is approximate for irregular sampling, and the plotted periodogram is NOT a
window-deconvolved intrinsic PSD. The frequency grid starts at 1/T and ends
at 0.30/day with spacing 1/(5*T); 0.30/day is a display ceiling, not a rigorous
irregular-sampling Nyquist frequency. Thin curves show individual ordinates;
thicker curves average power in 29 logarithmic bins. There are no slope fits,
periodicity claims, or independent-bin error estimates.

The blue dotted lower curve is the exact expectation of this same unfiltered
periodogram under the configured covariance and ACTUAL sample times, including
the floating mean. At each f, if P_f projects onto mean-removed sine/cosine columns,
it is (T/N)*trace(P_f K). It is not a fitted PSD or a fresh Monte Carlo draw.
Its difference from the top-panel intrinsic spectrum illustrates window and
sampling effects. The light-gray vertical line marks 1/T for T=458.511897 days.
No sampled periodogram is drawn below that line.

The source is sample-demeaned and has finite-realization RMS different from A.
Consequently the quoted ensemble long-period variance fractions and ideal
filter-retention fractions must not be confused with finite-sample RMS ratios.
PSD is in squared relative-flux units times days; no separate curve is normalized
to equal integrated power. High-frequency power in a sampled periodogram can
include leakage/aliasing of the much stronger low-frequency process.

Method reference: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
Reproduce with the astro environment and `plot_independent_slow_psd.py`.
Select existing cases with `--sigmas 5 10 20 40` and specify `--output-dir`
to retain the earlier two-sigma figures. The selected sigmas are in summary.json.
Orange: sigma5; dark blue: sigma10; red: sigma20; green: sigma40.
The royal-blue dotted curves are theory/reference, and the black curve is the original sample.
