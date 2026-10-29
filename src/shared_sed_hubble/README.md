# Shared standard-SED distance and H0 workflow

本模块统一六波段共同 SED、每源一个光度距离及下游 H0 分析。科学模型与绘图代码均在 src 中，运行时不导入 output 内的临时脚本。shared-hubble 保留固定零额外散布的源级基线；hubble-compare 接入 f_D 与本动速度的四方案比较。原来的 python main.py hubble 仍对应旧 log-distance 模型。

## 运行

在项目根目录、mica2环境中：

```bash
python main.py shared-hubble --background
```

默认十源、最多3源并行，结果位于`output/shared_sed_hubble`。后台进程独立运行；`launch.json`记录PID，`controller.log`记录总日志，`controller_status.json`记录逐源状态。重复运行会复用兼容的已完成源，失败源可继续；同时运行的控制器由输出目录锁区分。去掉`--background`即可在当前终端执行。

只重新画图、重新拟合H0、或复核已有结果：

```bash
python main.py shared-hubble --step plot
python main.py shared-hubble --step h0
python main.py shared-hubble --step validate
```

以上默认读取`output/shared_sed_hubble`；通过`--output`指定其他结果目录。例如复用本次基准结果，仅重画图：

```bash
python main.py shared-hubble --step plot --output output/gamma_relaxed_downstream_20260903/joint_hubble_shared_sed_no_f0
```

仅拟合或复用某个源使用`--source Mrk335`；只完成源级拟合而暂不汇总使用`--step sources`。`--step h0`只使用已完成的源距离，重新进行H0 MCMC；`--step plot`只读表格及H0样本，不调用FFA、不编译C库。

`defaults.yaml`固定本次源名单与输入分支。需要选择其他既有输入时，可复制该文件，修改`sources/result_subdir/output_dir/parallel`后用`--config 文件路径`。文件路径相对于项目根目录解析，`--root`可指定数据根目录。新源名单必须是一源一行，不能把同一AGN的多个epoch当成独立源。

## 固定的科学口径

- 输入：各源`runs/mica/gamma/phys12_width12_relaxed/2comp/result/comp0`中已生成的FFA去银河消光光变、lag后验及距离常数。该入口不运行MICA，不修改其先验、分量标签或detrend处理。
- 源内使用原FFA光变GP似然，包含两个行列式和各波段平坦offset先验的解析边缘化。标准共同SED为`R_lambda = R_ref (lambda/lambda_ref)^(-7/3)`，即`lambda R_lambda`斜率为`-4/3`。共同DRW时标与`R_ref`用MCMC采样。
- DRW与各波段R的原先验来自实际`fflux/run.yaml`；共同R区间取原各波段先验换算后的交集，仍在对数坐标均匀。完整保留原time_shifts。
- 参考波段offset按条件分布抽样。沿用现有流程的一条代表性X(t)重构，固定其亮暗极值；同一共享样本生成F_bright、epsilon。既有固定beta=4/3的tau0后验每次只取一次；lag与FFA仍采用阶段间独立近似。
- 倾角45度，距离常数和通量单位与本次基准一致。保留`D_L=(1+z)D_saved`口径。沿用正lag、`F_bright>F_faint>0`及`0<epsilon<1`条件，保留有效比例。同一组共享参数使用各波段公式所得D_L应相同。
- 每源距离均值及样本标准差（ddof=1）构成一项线性距离高斯似然。H0采用两组MCMC，均匀先验`[1,800] km/s/Mpc`，Omega_m=0.3、f0=0，不加本动速度，不增加额外散布，不按约化chi2放大误差。
- 源级及H0均用32 walkers、burn=1500，从6000步按自相关检查延长；保留长度/autocorr>50且ESS>1500。双种子及其源序号偏移沿用基准。更改源顺序会改变对应源的随机种子偏移。

这是分阶段推断。固定X极值、倾角、上游MICA光变重建及lag/FFA阶段关系的系统误差没有完整传播；每源D的高斯汇总也是近似。

## 代码和样式

- `inference.py`：原始似然的受限源级MCMC、距离传播、联合H0 MCMC、数值验证。
- `kernel.py` / `ffa_kernel.c`：与原FFA等价的加速内核。控制器在需要新源采样时显式编译，产物只写到结果目录`build/libffa_kernel.so`，worker只加载。使用已有cc/gcc，不安装依赖；导入或画图不会编译。
- `__main__.py` / `config.py` / `defaults.yaml`：入口、运行状态、缓存复用与路径配置。
- `style.py`：统一字体、颜色、全框、内向主次刻度、画布尺寸、源标签位置、Scott KDE显示带宽以及SH0ES/Planck参考区间。
- `plotting.py`：距离模数—红移及残差、联合H0后验，以及各源自由/共同SED比较。没有图外总标题或底部注释；PNG/PDF均保留。Mrk509空心符号继续标记其上游B波段低ESS。SH0ES/Planck只用于显示，不参与似然。

绘图也可在Python中复用：

```python
from pathlib import Path
from src.shared_sed_hubble.plotting import make_plots
make_plots(Path("output/shared_sed_hubble"))
```

## 输出与已知基准

`source_distance_summary.csv`保存每源D_L分位数、均值/标准差及诊断；`h0_summary.csv`和`h0_posterior.csv`保存联合结果；`fit_quality.csv`保存Hubble残差和SED比较；`sources/源名/`保存源配置、链和距离传播样本。

基准目录为`output/gamma_relaxed_downstream_20260903/joint_hubble_shared_sed_no_f0`。该试拟合得到H0=89.5662(-7.0347,+8.2732)，Hubble chi2/dof=49.6047/9。Mrk335、NGC6814的源间残差较大，多个源的标准SED失配明显，因此固定的是可复现的计算方法及样式，不能把该数值当作可靠的H0测量。NGC6814物理有效样本比例约88.12%；旧MICA中817 U/B、6814宽度及509 B低ESS的限制继续保留。

迁移验收保存在`output/shared_sed_hubble_migration_check`：复用完成的源级结果重新运行H0 MCMC和绘图，不重采样源。新旧C内核在真实NGC5548的固定参数点完全一致；新绘图模块对原表格重绘的两张PNG逐像素一致。`--step validate`独立检查距离公式、六波段等距、源ESS/双链分位数和H0的一维积分，对本次基准的积分/MCMC分位数差最大约0.0364 km/s/Mpc。

## 四方案本动速度比较

正式入口（同一 mica2 Python 环境）：

```bash
python main.py hubble-compare
python main.py hubble-compare --step validate
python main.py hubble-compare --step plot
python main.py hubble-compare --step plot --include-joint
```

默认读取已保存的 gamma 十源距离和 `data/peculiar_velocity/source_inputs.csv`；
结果独立保存到 `output/peculiar_velocity_comparison_20261029`。
`--step inputs` 只验证并合并输入，`--step fit` 完成推断及数值校验，
`--step all`（默认）继续绘图。`--config` 使用 `comparison.yaml` 格式，
与源级拟合的 `defaults.yaml` 区分；`--root` 解析输入路径，`--output` 指定结果目录。
此入口不重跑 FFA/MICA；源级距离生成仍使用 `shared-hubble`。

| 方案 | 额外相对距离散布 | 本动速度 |
| --- | --- | --- |
| A | 固定 f_D=0 | 关闭 |
| B | 拟合 f_D | 关闭 |
| C | 固定 f_D=0 | 流场均值 + 250 km/s 残余散度 |
| D（主方案） | 拟合 f_D | 流场均值 + 250 km/s 残余散度 |

四组共用同一份宿主系统 CMB 红移和原始距离均值/标准差，均传播 NED 已报告的
CMB 红移测量误差。`z_saved` 单独保留原距离计算所用红移；
它与新的 `z_cmb` 不互相覆盖。宿主/群选择在四组之间不得改变。
每个 AGN 只有一个距离和一个潜在径向速度，不能把各波段当作独立的本动速度观测。

取正速度为远离观察者。C/D 的模型为（v/c 的低速多普勒近似）：

$$
v_i \sim \mathcal N(v_{{\rm flow},i},250^2)\ {\rm km\,s^{-1}},\qquad
1+z_{{\rm cos},i}=\frac{1+z_{{\rm CMB},i}}{1+v_i/c}.
$$

平直 LCDM 中 `Omega_m=0.3`，光度距离预测和条件方差为：

$$
\mu_i=\frac{c(1+z_{{\rm cos},i})}{H_0}
\int_0^{z_{{\rm cos},i}}\frac{dz}{\sqrt{\Omega_m(1+z)^3+1-\Omega_m}},
\qquad
S_i^2=\sigma_{D,i}^2+(f_D\mu_i)^2.
$$

每源先对完整、含归一化项的
`Normal(D_mean | mu, S²)` 似然边缘化速度与红移误差，然后相乘；
不先平均距离，不把速度误差再加入 S²。
H0 的先验为 Uniform[1,800] km/s/Mpc；B/D 的 f_D 先验为尺度 1 的半正态，
数值网格到 5，并报告边界尾部质量。A/C 是独立的一维固定零散布模型。
积分通过独立正态红移/速度的比值分布化为一维，使用标准正态变量上的
Gauss-Legendre 求积，默认 64 节点；报告 z_cos<=0 的概率，
超过 1e-6 就报错，绝不静默截断明显的非物理质量。

`baseline_saved_z` 是额外的旧口径回归检查：采用原 z_saved、关闭新红移误差及
本动速度，并拟合 f_D。它应复现已有的 H0=72.692723、f_D=0.627168
中位数；新的 B 使用核实后的 CMB 红移，不应要求与旧值相等。

输出包括：

- `common_inputs.csv`、`input_provenance.json`：同一十源清单、原始/新红移、
  流场均值、250 km/s、出处及输入文件指纹。
- `comparison.csv`、`run_summary.json`：四组分位数、最大似然、先验与全部数值设置。
- `cases/A` 至 `cases/D`：H0 后验、源级红移区间与残差；
  B/D 另有 f_D 边缘和联合后验。
- `validation.json`：概率归一化、网格分辨率、求积加倍与边界质量检查。
- 图形 PDF 和 300 dpi PNG、`plot_manifest.json`：四面板哈勃图、H0 分布/区间、
  B/D 的 f_D 分布；可额外画联合后验。图外说明给出误差棒和区间定义。

哈勃图沿用原距离模数—对数红移图：对保存的 D_q16、D_q50、D_q84 分别计算
mu=5 log10(D_L/Mpc)+25，以 mu_q50 为点、mu_q16/q84 为非对称纵向区间。
每组附带 delta_mu 残差面板和原源名标注。该显示口径与原图一致；
推断仍使用线性距离均值/标准差，不改成模数空间拟合。f_D 不混入图上的距离分位区间。
横向区间是外部红移/速度分布的 68.27% 区间，没有条件于 H0 后验。
随红移增加变小的是相对误差 sigma_z/z，绝对 sigma_z 约为 (1+z)250/c，
不强制让高红移点的线性红移误差棒变短。

这版实现以保存的盘距离为条件，纳入本动速度对宇宙学红移的主要影响。
没有重新推断盘距离、修正多普勒增亮或传播跨源相干流场协方差；
独立 250 km/s 是统一工作假设，不是每源实测速度误差。
低红移流场必须落在固定模型覆盖范围内；缺失/未核实输入会停止计算。
已有 SED 失配、倾角与上游采样限制仍然存在，四组比较不自动消除这些问题。

实现位于 `redshift.py`（外部输入与约定）、`h0.py`（四组推断）、
`comparison_plotting.py`（只读结果绘图）及 `comparison.py`（统一控制器）。
原 `plotting.py`、`inference.py` 的历史入口继续兼容。

冻结输入快照遵循仓库既有 data 忽略规则。需要从保留的原始数据重建或校验时，
在具备 NumPy/Astropy 的环境中执行（主推断环境本身不需要 Astropy）：

```bash
python -m src.shared_sed_hubble.build_redshift_inputs --validate-only
python -m src.shared_sed_hubble.build_redshift_inputs
```

显式添加 `--refresh` 才重新查询当前 NED 并刷新固定版本的流场数据。
在全新数据目录首次重建时，也会下载尚不存在的原始输入。详细来源见
`data/peculiar_velocity/README.md`；主 H0 推断始终读取冻结快照，不联网。
