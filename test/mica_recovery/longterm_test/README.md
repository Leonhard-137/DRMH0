# Long-timescale injection and recovery

The experiment YAML selects the generator, cases, sampling, and MICA settings.
Run these commands in this directory with the existing `mica2` Python environment.

```bash
/home/leonhard/miniforge3/envs/mica2/bin/python prepare.py --config experiment_real_disk_mismatch.yaml --suite real_disk_mismatch
/home/leonhard/miniforge3/envs/mica2/bin/python plot_inputs.py --config experiment_real_disk_mismatch.yaml --suite real_disk_mismatch
/home/leonhard/miniforge3/envs/mica2/bin/python fit.py --config experiment_real_disk_mismatch.yaml --suite real_disk_mismatch --run --plot-results
```

For completed fits, run only the postprocessing entrypoint:

```bash
/home/leonhard/miniforge3/envs/mica2/bin/python plot_results.py --config experiment_real_disk_mismatch.yaml
```

`--case NAME` can be repeated; `--suite` and `--model gaussian|gamma|both` match
the fitting workflow. `--plots transfer lightcurves corner` selects figure types.
`--compare-input` adds an input-kernel overlay when its driver definition is
comparable. Example: `--config experiment_broad.yaml --suite known_response
--compare-input`. Input overlays use the saved native normalization factors;
they are not normalized to the recovered area or fitted to its height.

Each case keeps its input figures in `results/` and its recovery figures in
`results/gaussian/` and `results/gamma/`:

- `transfer_function.jpg/pdf`: black total median with 68.3% posterior shading;
  C0/C1 colored short dashes without shading; linear/logarithmic y panels.
  The lag window is the central 99.9% interval of the posterior-averaged,
  unit-area total kernels. Negative-lag area fractions are integrated over
  the full domain separately for every draw, then summarized by their medians.
- `light_curves.jpg/pdf`: W2 and U observations, total reconstruction and native
  errors, standardized residuals, and mean-subtracted U component reconstructions.
  Evidence is the original `ln Z`; residual RMS and available noiseless-input
  NRMSE use linear interpolation of the native reconstruction at observation
  times. These are in-sample diagnostics, not reduced chi-square or held-out scores.
  Component means are removed on the response observation epochs. Native
  reconstruction errors are shown without rescaling or adding component errors.
- `corner.jpg/pdf`: all equal-weight posterior draws, marginal histograms and
  two-dimensional count densities. Compact square panels follow the
  image-creator / Guo Fig. 16 layout: black solid step histograms, gray central
  68.3% strips, orange medians, and parameter summaries above each marginal.
  Solid black 2D contours mark highest-density thresholds targeting 68.3% and
  95.4% mass in the existing unsmoothed 28-by-28 count histogram. Tied bin
  counts can enclose more mass; interpolated contour lines are approximate,
  not Gaussian one/two-sigma ellipses. Coordinates with log10 use equal bins
  in that displayed coordinate. All samples and the original ranges remain.
  A and physical widths are plotted in log10;
  Gamma physical standard deviation is sqrt(2) times its native scale theta.
  Visible prior edges come from saved parameter metadata, not the current YAML.
- `summary.json`: numerical summaries, physical parameter bounds, normalization,
  evidence, area fractions, and available recovery metrics.

Both plotting entrypoints share `plot_style.py` and the project-local
`assets/image-creator.mplstyle`. The latter is copied from the approved skill;
there is no runtime dependency on the personal skill directory. Exact role
colors, gray capless errors, and three minor ticks per linear major interval
follow image-creator. Explicit project choices override its generic defaults:
Times New Roman instead of STIX, black total TF with colored short-dashed
components, and 300-dpi JPG plus vector PDF instead of PNG. Corner density
layers are rasterized; their text, axes and contours remain vector in PDF.
Probability definitions and normalization do not change with the style. Comparing evidence requires the same data and normalization and
depends on the selected priors.

Explanatory annotations are omitted by default, including corner upper-triangle
text, figure footnotes and methodological subtitles. Their definitions live in
this README, saved configurations/component tables and `summary.json`. Axes,
units, concise legends, band/model labels and corner parameter summaries remain.
The explicitly requested negative-lag fractions and light-curve recovery metrics
also remain visible. Input component curves marked `(+1)` are saved noise-free
variations offset by one for display; their amplitudes are not rescaled by the
plotter. Corner orange lines mark medians, gray strips the central 68.3% interval,
and gray dashed marginal lines the saved prior edges within the displayed range.

Transfer-function and corner labels are sorted by centroid within each joint
posterior draw: C0 is the smaller centroid and C1 the larger. Amplitude,
centroid and width move together; no samples are filtered. This is not a
width ordering or a disk/BLR identification. Saved native `pline` curves have
already been averaged and cannot be sorted this way; their light-curve legends
explicitly use Native 0/1. The summary records the relabelled draw count.
Use `--output-subdir results_centroid_order` to retain earlier native-label figures.

The native TF files can truncate at the lag-prior limits (for example 0--100 d).
The plot therefore reevaluates the same analytic kernels from the complete
posterior on an extended, peak-resolving grid. Components are summed within
each draw before total quantiles are calculated. The calculation is checked
against the saved native total and error bands on their original grid.

Truth comparisons are conditional: the known-response suite has an explicit
W2-to-U kernel; the independent suite has a known correlated-response kernel;
the spectral suite supplies an effective predictor. Physical I0-to-U kernels
and clean-driver pathways are not overlaid as total W2-to-U recovery truth.
For those suites no unique true fitted component parameter vector exists, so
corner figures show the posterior and priors without invented truth markers.
Noiseless total light curves are read from saved component tables when present,
including original tables reached through inference-only input symlinks.

Inverse-design code and its records live in [predict/](predict/prediction_readme.md).
The inference-only `experiment_shared_817prior.yaml` reuses existing inputs;
do not run `prepare.py` with that configuration.

## Paired one-shared-process controls

`experiment_ablation.yaml` selects five new matched controls. The sixth,
`cases/abl_disk_blr_shared`, links to the completed
`cases_one_shared_blr/one_shared_initial_refined` and is excluded from fitting.

| Case | BLR in U | Extra S in W2 | Extra S in U |
| --- | --- | --- | --- |
| `abl_disk_only` | off | off | off |
| `abl_disk_uonly` | off | off | on |
| `abl_disk_shared` | off | on | on |
| `abl_disk_blr` | on | off | off |
| `abl_disk_blr_uonly` | on | off | on |
| `abl_disk_blr_shared` (reuse) | on | on | on |

All controls retain the anchor's sampled illumination, shared process, epochs,
measurement noise and fixed alpha. I is a 30-day DRW; S retains the smooth
GP covariance `(1 + abs(h)/T) exp(-abs(h)/T)` with T = 55.3735295468 days.
S is independent of I; in U-only cases it is independent of W2. These controls
test the BLR/shared switches, not the DRW-versus-smooth-GP question.

The saved anchor contains combined disk+BLR illumination. `prepare_ablation.py`
draws the missing BLR subcurve conditional on the saved, band-demeaned W2/U
illumination under the original covariance. It retains the conditional residual
(seed 2026091902), subtracts this same BLR draw for all BLR-off cases, and never
rescales the individual cases. This is one compatible conditional extension,
not the unique historical hidden I realization. The generated component tables
retain their first five columns; two additional columns partition illumination
into disk and BLR and must not be added again to the total. Provenance and
conditional diagnostics are in each case's `config/paired_generation.json`.

```bash
python prepare.py --config experiment_ablation.yaml --suite paired_ablation
python plot_inputs.py --config experiment_ablation.yaml --suite paired_ablation
python fit.py --config experiment_ablation.yaml --suite paired_ablation --batch-size 10 --run --plot-results
```

Use the existing mica2 environment. Preparation refuses to overwrite inputs.
The fit command uses the existing checked resume/skip behavior. Its ten fits run
concurrently; after they finish, postprocessing runs sequentially into each
case's `results/gaussian` and `results/gamma`. For postprocessing alone:

```bash
python plot_results.py --config experiment_ablation.yaml --suite paired_ablation
```

## U-only GP without disk filtering

`experiment_uonly_gp_raw.yaml` creates `cases/abl_disk_blr_uonly_gp_raw`:
`W2=1+alpha*D_W*I+noise_W`,
`U=1+alpha*(a*D_U+b*B)*I+A_S*S+noise_U`.
The illumination and measurement noise are retained from the previous controls.
Only U receives the extra GP, directly at its observed epochs:
`C(h)=A_S^2*(1+abs(h)/T)*exp(-abs(h)/T)`,
`T=55.3735295468 d`, `A_S=0.15396862368414638`.
No disk filter or interband shift is applied to this GP.

`prepare_ablation.sample_unfiltered_gp` uses the experiment seed with
`spawn_key=(0,1)` and subtracts the U sample mean. This is a new GP draw,
not a deconvolution of the previously saved slow component. The first five
component columns retain the existing layout; the legacy
`shared_long_variation` column now holds the independent unfiltered U GP
(and is zero for W2). Generation metadata identifies this explicitly.
The anchor's fixed alpha and the original MICA priors are retained.

```bash
python prepare.py --config experiment_uonly_gp_raw.yaml --suite paired_ablation
python plot_inputs.py --config experiment_uonly_gp_raw.yaml --suite paired_ablation
python fit.py --config experiment_uonly_gp_raw.yaml --suite paired_ablation --batch-size 2 --run --plot-results
```

The two fits run concurrently; standard postprocessing then writes to this
case's `results/gaussian` and `results/gamma`.

