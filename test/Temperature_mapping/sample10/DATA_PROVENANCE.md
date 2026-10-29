# Main-ten temperature-mapping input provenance

These inputs preserve supplied flux values and measurement cadence. No additional dereddening is applied by this pipeline; upstream correction histories are not universally established. No MICA/FFA host-subtracted inputs are used. Flux and error adopt the same units with the evidence strengths below. Numerical validation does not establish the physical origin of map patterns.

## Physical-flux unit evidence

The adopted numerical unit is `1e-15 erg s^-1 cm^-2 Angstrom^-1` for flux and error. A documented unit is not an independent reconstruction of photometric calibration.

| Input | Evidence strength and boundary |
|---|---|
| Mrk817_a; NGC5548 | Reuse previously audited physical-flux inputs. Mrk817_a HST/Swift flux/error rows match the full reference; all NGC5548 arrays equal the audited input. |
| Mrk142 | Published machine-readable table explicitly specifies these physical units; the arbitrary mean-normalized Lijiang lV curve is excluded. |
| Mrk335 | Ground-table units explicitly verified. Swift adopts the existing project physical-flux convention; its calibration was not independently reconstructed in this run. |
| F9_a; Mrk279_b; NGC6814 | Public processed-data archive explicitly specifies these units; local-release identity is not certified row-for-row. NGC6814 six-band counts and first records match corrected record20430106, not an entire-array comparison. |
| Mrk509; NGC4151_a | Campaign/source identified; physical-flux units adopt the existing project convention, not an independently reconstructed calibration in this run. |
| NGC2617_a | Assumed project f_lambda units; original Vega zero-point/band conversion and whether prior dereddening occurred have NOT been reconstructed in this run. |

No additional dereddening is applied by this pipeline. This does not establish that every upstream source is uncorrected; absolute temperature amplitudes remain conditional on the stated unit/calibration assumptions.

## Executed acceptance checks

- All ten NPZ files loaded with `allow_pickle=False`; 22,870 measurements total. Finite values, positive errors, unique band names ordered by wavelength, ordered within-band times, valid integer band indices, and at least two distinct epochs per band passed.
- Every array of the NGC5548 NPZ is exactly equal to its audited original; no second error inflation.
- The analytic adjacent-triplet floor was independently compared with weighted-least-squares plus scalar root solving for 32 deterministic cases. Maximum absolute difference was 4.94e-13 in flux units.
- Mrk817_a's 2,454 HST/Swift records match the existing full reference exactly in flux and original error; matching, rather than an assumed half-day shift, sets the common THJD bounds.
- NGC6814 local counts and first time/flux/error records in every band match [corrected Zenodo record20430106](https://doi.org/10.5281/zenodo.20430106). This bounded verification is not an entire-array identity claim. Do not replace these with the erroneous older `NGC6814(All)` archive entries.
- Source time provenance and photometric calibration limitations below are not certified away by numerical validation. In particular, NGC2617_a inherits the project's already-converted physical-flux convention; the original Vega-zero-point conversion has not been independently reconstructed.

## Data inventory

| Source | Bands | Points | Span (d) |
|---|---:|---:|---:|
| Mrk817_a | 17 | 6135 | 420.101 |
| Mrk142 | 11 | 2308 | 230.820 |
| Mrk335 | 10 | 2457 | 185.295 |
| F9_a | 6 | 1863 | 708.588 |
| Mrk279_b | 6 | 1034 | 172.380 |
| Mrk509 | 6 | 1409 | 272.642 |
| NGC5548 | 19 | 4203 | 288.974 |
| NGC4151_a | 6 | 1693 | 69.254 |
| NGC6814 | 6 | 1511 | 74.760 |
| NGC2617_a | 6 | 257 | 49.650 |

## Frozen preprocessing

- NK22 arithmetic MEAN of nonnegative extra standard deviations from adjacent three-point straight-line chi2=1 fits; added in quadrature.
- Mrk817_a: exact matching determines published-time campaign endpoints; original errors are re-expanded using the mean floor. Full817 reference remains unchanged.
- NGC5548: every NPZ array is verified identical to the existing audited local_input.npz; no reinflation.
- Other sources: local MJD and physical-flux convention are retained. No unknown UTC/TDB/heliocentric correction is invented.
- X-ray bands and Mrk142's uncalibrated Lijiang lV are omitted. Nonidentical same-time rows are retained and flagged.
- Per-band cadence/gaps, exact paths, wavelength provenance, error floors and exclusions are in derived/input_manifest.json and per-source metadata.

## Mrk817_a

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/Mrk817_a_input.npz`

Time: THJD=published HJD-2450000, observer-frame days

Selection: _a endpoints identified by exact flux/error matching of all local HST/Swift rows, then applied as a common published-THJD window to all17 bands

- This is Mrk817_a (about420 d), not the full524-day reference. The original full817 reproduction is unchanged.
- New errors use NK22 mean floors, whereas the full817 reference used NK24 median floors. Compare with awareness of both changes.
- Time bounds use matched published HJD, NOT an assumed MJD+0.5 conversion.

## Mrk142

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/Mrk142_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- Lijiang lV (62 rows) excluded: arbitrary flux normalized to mean 1, NOT physical f_lambda (Cackett2020 Table2).
- Ground r/i/z counts differ from paper Table1; this is the supplied local screened release, not an exact published selection.
- Swift wavelength convention follows project config (M2=2246,U=3465 A); paper Table3 gives 2236,3467 A.
- Ground monitoring extends beyond Swift; those epochs have weaker radial information.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://arxiv.org/html/2005.03685

## Mrk335

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/Mrk335_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- Swift physical-flux units adopt the existing project convention; Swift calibration was not independently reconstructed in this run. Ground-table physical-flux units were explicitly verified.
- Ground griz are the supplied local intercalibrated release; counts and some flux/error values differ from Kara2023 published Table2.
- Time origin is retained as local MJD. Cross-instrument UTC/TDB and heliocentric-conversion provenance is not fully documented. Published ground-table comparisons show telescope-dependent offsets; sub-day propagation interpretation needs a separate timing audit.
- z combines z-prime/z_s; 8700 A is the paper Table3 effective-wavelength approximation, not a measured transmission kernel.
- Ground monitoring extends beyond Swift; those epochs have weaker radial information.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Ground_i retains 2 same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.
- Ground_z retains 1 same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.

Primary source: https://arxiv.org/html/2302.07342

## F9_a

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/F9_a_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- Late Fairall9 campaign only; local screened/cropped release differs from Wang2025 segment2 and is NOT the early Fairall9 dataset mapped in NK22.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://doi.org/10.5281/zenodo.14930797

## Mrk279_b

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/Mrk279_b_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- Segment2 only; no data from Mrk279_a or segment3 are merged.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://doi.org/10.5281/zenodo.14930797

## Mrk509

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/Mrk509_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- 2017 Swift campaign; six UVOT bands only in this local source table.
- Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://arxiv.org/abs/1811.07956

## NGC5548

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/NGC5548_input.npz`

Time: THJD=HJD-2450000, geocentric local MJD_UTC plus builtin heliocentric correction

Selection: Full audited 2014 STORM campaign including HST/ground coverage outside the shorter Swift window

- Later public5900-point TempMap bundle is NOT used: it mixes Swift MJD and HST/ground HJD and includes another Swift release.
- This input is4203 original observed points; no extinction or host correction.
- Public wavelength vector is inherited; NK22 text lists slightly different U/u/z wavelengths. This is already tested in the individual5548 reproduction.

## NGC4151_a

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/NGC4151_a_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- 2016 Swift campaign only; no later NGC4151_c data are merged.
- Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://arxiv.org/abs/1811.07956

## NGC6814

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/NGC6814_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- No additional dereddening is applied by this pipeline despite substantial Galactic reddening; upstream correction history is not independently established, and temperature amplitudes are conditional on the supplied-flux convention.
- Gonzalez2024 correction (stae1353) changes UV calibration and withdraws the extreme blue-spectrum/truncation conclusion of the original paper.
- Zenodo14930797 has erroneous NGC6814 bands. Local counts and first time/flux/error rows of ALL six bands match corrected record20430106, Lightcurves_new/NGC6814(All); entire-array equality was not checked. No re-calibration or detrending applied.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://doi.org/10.1093/mnras/stae1353

## NGC2617_a

Input: `/home/leonhard/sci001/test/Temperature_mapping/sample10/derived/NGC2617_a_input.npz`

Time: Original local MJD column retained, observer-frame days; no origin shift

Selection: entire source-specific raw CSV, without merging similarly named campaigns

- 2013 outburst Swift subset only (about50 d); no later NGC2617 campaign is merged.
- Assumed project f_lambda units; original Vega zero-point/band conversion and whether prior dereddening occurred have NOT been reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.

Primary source: https://arxiv.org/abs/1310.2241
