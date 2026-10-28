# Shared standard-SED distance and H0 workflow

本模块固定2026-09-04已运行的六波段共同SED、每源一个光度距离、固定f0=0的分析。科学模型与绘图样式均在src中；运行时不导入任何output目录内的临时脚本。原来的`python main.py hubble`仍对应旧的log-distance层次模型，本模块入口为`shared-hubble`。

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
