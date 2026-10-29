# Tau2, disk:BLR area 9:1 recovery

W2=baseline+alpha*DW*I+noise; U=baseline+alpha*(0.787*DU*I+(0.787/9)*B*I)+S+noise; S is independent and directly added only to U.

Unit stationary RMS latent DRW; fixed alpha=1.45; no disk/BLR sample rescaling. Preserved slow realization scaled to sample RMS 1.3. Common additive baseline ensures positive flux. Baseline=7.0.

Five matched cases: raw and BOTH-band sigma5/10/20/40 detrending. Two Gaussian and two Gamma fits per case; 10 simultaneous fits, 2 MPI ranks each; max_num_saves=10000, ptol=0.01. Full filtered noise covariance saved; original diagonal errors used by MICA. Plot transfer functions, corner and light-curve diagnostics serially after fitting. TF/corner labels sorted by centroid per draw.
