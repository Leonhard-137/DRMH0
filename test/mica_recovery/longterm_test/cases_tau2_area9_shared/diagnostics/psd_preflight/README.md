# Shared-S PSD preflight

Status: **PASS**

Input is never edited and this program does not run MICA.

## What is plotted

- `raw_components_psd`: theoretical central/slow PSD and fP, plus raw saved W2/U component periodograms.
- `six_case_component_psd`: all six saved cases, same axes, full light curve and known disk/BLR/S/noise components.
- `slow_filter_psd`: saved S spectra and exact expected LS spectra under the specified covariance at each actual sample window after the actual discrete filter.

PSD uses unweighted floating-mean LombScargle, multiplied by 2T/N. Logarithmic bins show arithmetic mean periodogram power. No interpolation, no separate unit-power normalization, no line search, and no fitted spectral slope is used. Upper frequency 0.30/day is a display ceiling, not an irregular-sampling Nyquist frequency. Component PSDs do not add exactly in one realization because cross terms exist.

Expected LS spectrum is (T/N) trace(P_f K_filtered), with P_f the projector onto centered sine/cosine columns; K_filtered=F K F^T. This explicitly retains the observing window and finite-sample filter effects. Intrinsic fP peaks outside the observing baseline cannot be read back as resolved measured peaks.

Hard gates: all inputs finite and positive; approved parameters; original signal/noise/times/errors preserved; exact component sum and actual filter operator; shared conditional-GP extension; valid finite spectra; <0.01/day S power attenuated and correctly ordered with sigma; sigma5 leaves <10% of raw low-band power; raw W/U low-band S power ratio within factor two of the same-process window expectation. Numerical tolerances and all measured values are in report.json. Scientific modeling limitations below are explicitly not classified as unexpected generation defects.

## Warnings

- The 458-day observation window is shorter than the S fP(f) peak period (~603 days); this peak cannot be resolved from these saved light curves.
- LombScargle density and its frequency-band integrals are descriptive, window-dependent quantities for irregular sampling; they are not intrinsic spectral variance fractions. High-frequency leakage of S is not by itself evidence of a faulty generator.
- tau_I=2 days implies only ~43% of theoretical driver variance at periods <10 days. This is the approved model, not an unexpected PSD failure.
- The unchanged W2/U observation times differ. Identical sigma therefore gives slightly different discrete filters; filtered S is not exactly identical at different epochs.
- High-pass filtering changes the driver covariance away from a DRW and correlates errors. Native MICA still uses its existing DRW model and quoted diagonal errors; this preflight does not validate that likelihood approximation.
- u_s20 intentionally filters only U. Its effective W2-to-U kernel differs from the unfiltered physical response; it is a control, not a direct physical-kernel recovery measurement.
- raw: residual U S/BLR RMS ratio=32.6; attenuation need not yield successful BLR recovery.
- all_s5: U BLR RMS 0.0085861 is below realized noise RMS 0.0269222; weak BLR identifiability is an expected scientific limitation, not a PSD-generation anomaly.
- all_s5: residual U S/BLR RMS ratio=5.55; attenuation need not yield successful BLR recovery.
- all_s10: U BLR RMS 0.0170512 is below realized noise RMS 0.0279587; weak BLR identifiability is an expected scientific limitation, not a PSD-generation anomaly.
- all_s10: residual U S/BLR RMS ratio=6.8; attenuation need not yield successful BLR recovery.
- all_s20: U BLR RMS 0.0277338 is below realized noise RMS 0.0288999; weak BLR identifiability is an expected scientific limitation, not a PSD-generation anomaly.
- all_s20: residual U S/BLR RMS ratio=10.4; attenuation need not yield successful BLR recovery.
- all_s40: residual U S/BLR RMS ratio=20.2; attenuation need not yield successful BLR recovery.
- u_s20: U BLR RMS 0.0277338 is below realized noise RMS 0.0288999; weak BLR identifiability is an expected scientific limitation, not a PSD-generation anomaly.
- u_s20: residual U S/BLR RMS ratio=10.4; attenuation need not yield successful BLR recovery.

## Blocking anomalies

