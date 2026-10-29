# Recovery fits for six saved PSD candidate realizations

Each selected candidate contributes raw and BOTH-band sigma5/10/20/40 Gaussian high-pass inputs. These are the exact previously selected single realizations, imported without redrawing or RMS rescaling. Absolute flux is the saved fractional variation plus one, times the original observed band mean. The original MJD epochs and absolute measurement errors are retained.

The raw candidate spectra are checked against saved examples. Sigma20/40 are additionally checked against their existing filtered spectra. Every filter is F-W@F+mean(F), applied independently to both actual band grids; row normalization need not preserve the output arithmetic mean. Full propagated noise covariance is saved, but native MICA continues to use original diagonal errors.

Physical kernel truth and source seed/index are retained in generation.json. The selected NPZs only contain total noisy curves; component light curves are not fabricated. Hence existing plot_results.py produces inferred transfer functions, corner and light-curve plots without a false noiseless-truth overlay.

Candidates: C045, C001, C078, C075, C089, C069. Cases: 30. Fits: 60. Each fit has two components, max_num_saves=10000, ptol=0.01, two MPI ranks, original priors. The separate controller is limited to eight concurrent fits, prioritizes Gamma, and serializes post-fit plots. This preparation script launches no sampler. Existing inputs/results are never overwritten.
