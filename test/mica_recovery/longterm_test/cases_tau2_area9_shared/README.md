# Tau2, area 9:1, same direct slow process in W2 and U

W2=7+alpha*DW*I+S+noise; U=7+alpha*(0.787*DU*I+(0.787/9)*B*I)+S+noise; S independent of I, same direct process at both bands, gain=1, lag=0, no convolution

Six matched inputs: raw; BOTH-band sigma5/10/20/40 detrending; U-only sigma20. Each receives two-Gaussian and two-Gamma fits (12 fits, 2 MPI ranks each). Priors and sampling retain max_num_saves=10000, ptol=0.01.

Only W2 is given the additional S. U, disk, BLR, sampling, original errors and noise remain paired with cases_tau2_area9. S is independent of I and common to the two bands. Its original sample is available only at U epochs: a conditional Matérn3/2 GP draw extends the same sample to W2 epochs, retaining the residual covariance. Both bands' S are separately sample-centered (constant zero-point shifts). This does not introduce a relative lag, distinct random S, gain change, or disk convolution.

Conditional evidence is in shared_slow_extension.npz and manifest.json. Baseline stays 7. Full filtered noise covariance is saved, while MICA retains original diagonal errors. Input generation does not run MICA: launch is conditional on a separate PSD preflight.
