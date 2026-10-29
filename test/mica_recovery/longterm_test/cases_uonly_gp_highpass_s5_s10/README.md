# Paired U-only high-pass sigma5/10 experiment

Input realization: ../cases/abl_disk_blr_uonly_gp_raw. No fresh random draw.
W2 is copied unchanged. U = original - Gaussian weighted trend + original mean,
with sigma=5 or 10 observer-frame days; all component columns receive the same
linear operation. Epochs and quoted errors are retained. The full transformed
noise covariance is saved for reference; MICA keeps the original diagonal errors.

Configuration: ../experiment_uonly_gp_highpass_s5_s10.yaml (inference only).
Prepare reproducibly with ../prepare_uonly_highpass_s5_s10.py, which refuses
to overwrite existing outputs. Do not use the generic prepare.py for this config.
Sampling and priors match the sigma20/40 control: two components, positive
amplitudes, centroids [0,100] d, max_num_saves=10000, ptol=0.01, two MPI ranks
per fit, one particle per rank. Gaussian sigma is [0.14195358,152.84] d;
Gamma physical standard deviation is [0.20075267806341413,216.14840087310387] d.

run_pipeline.sh launches all four case/model fits concurrently through the
existing fit.py --batch-size 4 runner, then serially produces transfer functions,
light-curve diagnostics and corner plots. Transfer-function and corner component
labels are ordered by centroid within each posterior draw: C0 <= C1. Saved
native component light curves retain their explicit Native 0/1 labels.

Results live in each case's results/{gamma,gaussian}/. Detailed native per-fit
logs and run_mica_state.json are in each case's runs directory. pipeline.log
records the outer scheduler; pipeline_exit_code records the final command status.
