# 主要十源：温度映射与慢／快变化复现测试

实际完成 10/10 源，保存 134 组拟合结果；失败 0 源。

样本严格来自 `DRMH0/tables/sample_observations.csv`；不同后缀是特定观测期，不将同一 AGN 当作多个独立源。完整 Mrk817 与此前 NGC5548 的独立复现结果保留在上级目录。

## 这一轮首先看到了什么

- 在本轮固定模型和比较区域下，28 天低通温度场的 RMS／原始温度场 RMS 为：Mrk817_a 0.838、F9_a 0.850、Mrk279_b 0.814、Mrk509 0.877；NGC5548 为 0.582。这只是拟合场的慢变化保留程度，不是慢波存在的概率，也不能作为跨源物理强度排名。
- 在四个具有额外 HST／地面波段的源中，仅 Swift 六波段图与全波段图的相关为 0.411–0.638。全波段约束确实影响恢复图样；另外六个源本来只有 Swift，表中的 1.000 是相同输入的恒等比较。
- 去掉 U/u 后，NGC5548 的图相关为 0.752，Mrk817_a 为 0.927；它们对这些波段的敏感性不同，但这里尚未分离 BLR。
- NGC4151_a、NGC6814、NGC2617_a 的总时长只有约 69、75、50 天；Mrk142 和 Mrk335 的全波段共同时间包络也只有约 106、99 天。56 天滤波结果都需要特别谨慎。

这些是首轮条件性结果，尚未进行传播轨迹显著性或不同物理机制的注入检验。

## 先看哪些文件

- `outputs/temperature_mapping_atlas.pdf`：全部图的 PDF 图册。
- `outputs/figures/`：每幅图的矢量 PDF、300 dpi PNG 与 JPEG 预览。
- `outputs/summary.csv`：逐源主拟合数值。
- `outputs/<source>/fits.npz`：原始数据、预测、径向核及每种测试的温度场。
- `outputs/<source>/metrics.json`、`summary.json`、`band_residuals.json`：各测试指标、条件和逐波段残差。
- `DATA_PROVENANCE.md`、`derived/input_manifest.json`：数据、单位、时标、误差修正与缺口。
- `PARAMETERS.md`、`physical_parameters.json`：固定盘参数和文献来源。

## 冻结的分析方法

在给定薄盘 $T_0(R)$、黑洞质量、距离、倾角和吸积率的条件下，用线性化黑体核将 $q=\delta T/T_0$ 积分为多波段光变；每个波段另有一个不受惩罚的常数偏置。最小化带观测误差权重的残差平方和与 NK22 型幅度、径向及时间差分惩罚之和。

主网格为 50 个径向格、250 个时间格，主正则化 ξ=10；ξ=1、100、1000 为固定对照，不按图样好看程度或理论预期选择。径向范围依源的文献设置确定，因此跨源相同格数也不代表同一径向分辨率。

| 测试 | 操作 | 能检查什么 |
|---|---|---|
| 原始光变 | ξ=1/10/100/1000 | 正则化敏感性 |
| 低通 | 每个波段先插值到 0.1 天网格，再用 FWHM=7/28/56 天高斯平滑，保留原始权重反演 | 慢变化对应的正则化温度场 |
| 高通 | 原始光变减去对应的低通结果，再反演 | 去掉该慢变化后剩余图样 |
| 时间网格 | ξ=10，Nt=100/250/500 | 离散化及惩罚变化的联合敏感性，不等于严格收敛检验 |
| 去掉 U/u | 排除中心波长 3200–3800 Å 的波段，保持相同 R、t 网格和 ξ | 这些波段对结果的影响，不证明其他波段无 BLR 污染 |
| 仅 Swift 六波段 | 唯一识别六滤光片后重拟合；本来仅六波段时直接标记与主拟合相同 | 增加其他波段是否改变图样 |

没有把 MICA 分解结果当作输入；没有联合 MICA 与温度场，也没有对十源运行残差时延 MCMC、传播速度拟合或检出显著性标定。

## 逐源主结果

下表 q99 是全图 |δT/T| 的第 99 百分位，不是误差条、显著性或盘变动的独立检测统计量。χ²/N 是每个数据点的加权残差平方，不是除以自由度的约化 χ²。

| 源 | 波段 | 数据点 | 总跨度/d | Δt/d | χ²/N | q99 |
|---|---:|---:|---:|---:|---:|---:|
| Mrk817_a | 17 | 6135 | 420.1 | 1.687 | 1.114 | 0.163 |
| Mrk142 | 11 | 2308 | 230.8 | 0.927 | 0.771 | 0.081 |
| Mrk335 | 10 | 2457 | 185.3 | 0.744 | 1.169 | 0.201 |
| F9_a | 6 | 1863 | 708.6 | 2.846 | 0.600 | 0.127 |
| Mrk279_b | 6 | 1034 | 172.4 | 0.692 | 0.414 | 0.120 |
| Mrk509 | 6 | 1409 | 272.6 | 1.095 | 0.617 | 0.145 |
| NGC5548 | 19 | 4203 | 289.0 | 1.161 | 1.370 | 0.174 |
| NGC4151_a | 6 | 1693 | 69.3 | 0.278 | 0.406 | 0.062 |
| NGC6814 | 6 | 1511 | 74.8 | 0.300 | 0.347 | 0.089 |
| NGC2617_a | 6 | 257 | 49.7 | 0.199 | 0.375 | 0.106 |

## 稳定性与 28 天分解

相关系数只描述选定比较区域的两张拟合图是否相似，没有传播方向或显著性含义。通常使用所有波段时间包络的交集以及平均归一化径向核累计 5%–95% 的径向格；该区域仍可包含观测缺口，也不等于有效分辨率或置信区间。若交集不足则回退全图并在单源 JSON 中明示。

低／高通 RMS 比是对应温度场与原始拟合温度场的 RMS 之比；两者相关，不能解释成可相加的物理功率占比。

| 源 | 去 U/u 图相关 | 仅 Swift 图相关 | Nt=100 图相关 | Nt=500 图相关 | 低通28 RMS比 | 高通28 RMS比 |
|---|---:|---:|---:|---:|---:|---:|
| Mrk817_a | 0.927 | 0.632 | 0.895 | 0.918 | 0.838 | 0.462 |
| Mrk142 | 0.901 | 0.638 | 0.801 | 0.910 | 0.507 | 0.851 |
| Mrk335 | 0.964 | 0.571 | 0.851 | 0.938 | 0.587 | 0.675 |
| F9_a | 0.991 | 1.000 | 0.949 | 0.971 | 0.850 | 0.375 |
| Mrk279_b | 0.980 | 1.000 | 0.918 | 0.985 | 0.814 | 0.476 |
| Mrk509 | 0.966 | 1.000 | 0.955 | 0.979 | 0.877 | 0.328 |
| NGC5548 | 0.752 | 0.411 | 0.892 | 0.890 | 0.582 | 0.663 |
| NGC4151_a | 0.825 | 1.000 | 0.750 | 0.861 | 0.341 | 0.876 |
| NGC6814 | 0.948 | 1.000 | 0.904 | 0.916 | 0.679 | 0.712 |
| NGC2617_a | 0.983 | 1.000 | 0.907 | 0.968 | 0.427 | 0.653 |

## 逐源限制与需要优先检查的点

### Mrk817_a

全波段时间包络交集为 397.6 天；最大单波段缺口 51.4 天。所有时间宽度均为观测者系。

- This is Mrk817_a (about420 d), not the full524-day reference. The original full817 reproduction is unchanged.
- New errors use NK22 mean floors, whereas the full817 reference used NK24 median floors. Compare with awareness of both changes.
- Time bounds use matched published HJD, NOT an assumed MJD+0.5 conversion.
- The Mrk817_a source label/selected segment is not by itself evidence that its input rows equal the full 17-band paper campaign; data selection is determined separately.
- At least one band has a gap >=28 days; Gaussian filtering interpolates across gaps.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### Mrk142

全波段时间包络交集为 105.5 天；最大单波段缺口 20.9 天。所有时间宽度均为观测者系。

- Lijiang lV (62 rows) excluded: arbitrary flux normalized to mean 1, NOT physical f_lambda (Cackett2020 Table2).
- Ground r/i/z counts differ from paper Table1; this is the supplied local screened release, not an exact published selection.
- Swift wavelength convention follows project config (M2=2246,U=3465 A); paper Table3 gives 2236,3467 A.
- Ground monitoring extends beyond Swift; those epochs have weaker radial information.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Super-Eddington accretion invalidates a strict radiatively efficient thin-disk interpretation; treat the recovered map as a conditional formal benchmark.
- Local standard-lag benchmark M=1.95e6 Msun and L_bol/L_Edd=3.40 uses a different literature compilation; it is intentionally not substituted for the NK22 mapping parameters.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### Mrk335

全波段时间包络交集为 99.2 天；最大单波段缺口 15.1 天。所有时间宽度均为观测者系。

- Swift physical-flux units adopt the existing project convention; Swift calibration was not independently reconstructed in this run. Ground-table physical-flux units were explicitly verified.
- Ground griz are the supplied local intercalibrated release; counts and some flux/error values differ from Kara2023 published Table2.
- Time origin is retained as local MJD. Cross-instrument UTC/TDB and heliocentric-conversion provenance is not fully documented. Published ground-table comparisons show telescope-dependent offsets; sub-day propagation interpretation needs a separate timing audit.
- z combines z-prime/z_s; 8700 A is the paper Table3 effective-wavelength approximation, not a measured transmission kernel.
- Ground monitoring extends beyond Swift; those epochs have weaker radial information.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Ground_i retains 2 same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.
- Ground_z retains 1 same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.
- Mass and Eddington ratio follow the project's documented compilation, not the distinct parameters used for the theoretical lag example in Kara et al. 2023.
- The luminosity distance is cosmology-assumed, with no peculiar-velocity or flow correction.
- 99th percentile |deltaT/T| >0.2: inspect the small-perturbation approximation; 0.2 is a descriptive flag, not a validated rejection threshold.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### F9_a

全波段时间包络交集为 708.6 天；最大单波段缺口 21.7 天。所有时间宽度均为观测者系。

- Late Fairall9 campaign only; local screened/cropped release differs from Wang2025 segment2 and is NOT the early Fairall9 dataset mapped in NK22.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Local F9_a is a later Swift segment (MJD 58550.52-59259.11), not NK22's earlier Swift plus optical campaign: this is method transfer.
- The local late-campaign benchmark uses M=2.60e8 Msun and lambda_Edd=0.035; NK22's older normalization is deliberately retained and is not asserted to be a simultaneous luminosity estimate.
- 50 radial cells over [1,400] do not exactly reproduce NK22's trimmed radial discretization.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### Mrk279_b

全波段时间包络交集为 170.9 天；最大单波段缺口 13.2 天。所有时间宽度均为观测者系。

- Segment2 only; no data from Mrk279_a or segment3 are merged.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Historical mass/accretion inputs are transferred to the local Swift segment 2, not refitted to its luminosity state.
- The luminosity distance is cosmology-assumed, with no peculiar-velocity or flow correction.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### Mrk509

全波段时间包络交集为 271.6 天；最大单波段缺口 6.4 天。所有时间宽度均为观测者系。

- 2017 Swift campaign; six UVOT bands only in this local source table.
- Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- Local standard-lag benchmark uses M=1.1e8 Msun and lambda_Edd=0.095 from a later study; NK22's 0.05 is deliberately retained for mapping.
- 50 radial cells over [1,400] do not exactly reproduce NK22's trimmed radial discretization.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### NGC5548

全波段时间包络交集为 123.9 天；最大单波段缺口 14.0 天。所有时间宽度均为观测者系。

- Later public5900-point TempMap bundle is NOT used: it mixes Swift MJD and HST/ground HJD and includes another Swift release.
- This input is4203 original observed points; no extinction or host correction.
- Public wavelength vector is inherited; NK22 text lists slightly different U/u/z wavelengths. This is already tested in the individual5548 reproduction.
- The local Swift light curves are the 2019 reprocessing, not automatically identical to the input version used by NK22.
- The separate local standard-lag benchmark uses logM=7.72 and lambda_Edd=0.05; the existing NK22 mapping values are deliberately preserved.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### NGC4151_a

全波段时间包络交集为 69.2 天；最大单波段缺口 1.2 天。所有时间宽度均为观测者系。

- 2016 Swift campaign only; no later NGC4151_c data are merged.
- Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- The local later standard-lag benchmark adopts M=2.30e7 Msun and lambda_Edd=0.014; those are not substituted for NK22's mapping inputs.
- 50 radial cells over [1,400] do not exactly reproduce NK22's trimmed radial discretization.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### NGC6814

全波段时间包络交集为 74.8 天；最大单波段缺口 3.2 天。所有时间宽度均为观测者系。

- No additional dereddening is applied by this pipeline despite substantial Galactic reddening; upstream correction history is not independently established, and temperature amplitudes are conditional on the supplied-flux convention.
- Gonzalez2024 correction (stae1353) changes UV calibration and withdraws the extreme blue-spectrum/truncation conclusion of the original paper.
- Zenodo14930797 has erroneous NGC6814 bands. Local counts and first time/flux/error rows of ALL six bands match corrected record20430106, Lightcurves_new/NGC6814(All); entire-array equality was not checked. No re-calibration or detrending applied.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- This transfers the mass, distance and fiducial luminosity normalization, not Gonzalez et al.'s relativistic irradiated-disk model: their i=45 deg, spin and color correction are not implemented.
- Published lag/SED behavior disfavors a self-consistent standard-disk explanation; map structures remain conditional on the deliberately simplified kernels.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

### NGC2617_a

全波段时间包络交集为 49.7 天；最大单波段缺口 4.3 天。所有时间宽度均为观测者系。

- 2013 outburst Swift subset only (about50 d); no later NGC2617 campaign is merged.
- Assumed project f_lambda units; original Vega zero-point/band conversion and whether prior dereddening occurred have NOT been reconstructed in this run.
- Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.
- UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.
- The Eddington ratio is an approximate epoch scaling with fixed bolometric correction, not a new fit to the 2013 SED.
- A changing-look outburst need not satisfy the steady thin-disk baseline assumption.
- The luminosity distance is cosmology-assumed, with no peculiar-velocity or flow correction.
- Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.
- Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.
- Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.
- All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.

比较区域：all-band common window, central 90% cumulative mean normalized kernel。

## 验证与计算成本

独立小规模稠密解与生产稀疏解的温度最大差为 1.53e-12；新增可选径向起点／固定时间边界在默认设置下产生的前向矩阵与旧版完全相同。

十源检查包括正常方程残差、固定偏置不改变温度场，以及每个平滑宽度下 q_low+q_high=q_original 的数值线性恒等式；上述温度／预测检查中的最大绝对偏差为 2.32e-11。线性恒等式验证实现，不是物理独立证据。

顺序单线程拟合累计 187.6 秒（不含资料核查、绘图和人工检查）；逐源计时保存在 summary.csv。

## 解释边界

1. **得到条纹不等于检测到传播。** 这里的观测量是多波段积分光度；温度场仍是欠定反问题的正则化解。未提供像素后验或不同传播模型的校准检验。
2. **平滑改变噪声协方差。** 按原文对照保留原始误差作为权重，因此低／高通 χ² 仅作描述，不能用独立高斯似然显著性解读。边界采用常值延拓，缺口用线性插值。
3. **时间跨度不同。** 短于 112 天的监测甚至不能覆盖两个 56 天平滑宽度；没有长条纹不能作为慢变化不存在的证据。
4. **ξ 相同不代表相同物理先验强度。** 数据误差、波段核、半径尺度和 Δt 均不同；Nt 扫描也改变未按物理导数归一化的差分惩罚。
5. **固定物理模型不等于真实盘。** 质量、距离、吸积率、倾角以及未统一去消光等选择影响径向核和振幅；超爱丁顿目标的薄盘解应视为形式测试。没有把这些不确定性传播到图中。本轮不额外去消光；尤其 NGC2617_a 的上游星等转流量及是否预先去红化尚未独立重建，绝对温度幅度只能在本地单位约定成立的条件下解释。
6. **U/u 去除不构成 BLR 分离。** 共模或宽光谱的 BLR／其他非盘变化仍可被温度模型吸收。
7. **这是方法迁移与数据条件下的复现测试。** 不是声称每个观测期都有一篇完全相同输入的发表温度图可供逐像素复现。

## 方法来源

- [Neustadt & Kochanek 2022](https://arxiv.org/abs/2201.10565v2)：线性温度反演与多源应用。
- [AGN STORM 2 VI](https://arxiv.org/html/2310.01497)：7/28/56 天平滑与相减对照。
- 本次数据和盘参数的逐项来源见同目录 DATA_PROVENANCE.md、PARAMETERS.md。
