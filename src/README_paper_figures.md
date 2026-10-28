# 论文绘图流程

## 全样本与论文插图

从项目根目录、在 `astro` 环境中运行：

```bash
python src/render_paper_result_tables.py
python src/render_paper_figures.py --stage mica_gamma2_physw_relaxed --reuse-ffa
cd DRMH0
latexmk -xelatex -interaction=nonstopmode -halt-on-error sample701.tex
```

`render_paper_result_tables.py` 从已有松先验结果更新 `DRMH0/tables/relaxed/` 的数值表、来源清单和联合结果宏。U 超额由保存的 FFA `R` 联合后验重算；不启动 MICA、FFA 或 H0 采样。主距离表与哈勃图共同读取 `joint_hubble_shared_sed_no_f0`，附录逐波段距离仅作为诊断。

样本及顺序读取 `DRMH0/tables/sample_observations.csv`；每个源有三类单源图，全部使用已保存的结果。MICA 和 SED／lag 默认使用 `mica_gamma2_physw_relaxed`，缺失该分支会报错，不会退回其他结果。绘图不会启动新的拟合。`--reuse-ffa` 复用已有 FFA 单源图，仅按当前样本重新拼附录页；需要重画这些单源图时去掉该选项。

本次论文使用十个具有完整松先验结果的源，共 30 张单源图；SED／lag 每行两个源、四个面板，十个源合为一页。FFA 附录同样每行两个源，按 6＋3 源分为两页。窄版面板按最终尺寸重绘文字，保留全尺寸单源图。原样本的 NGC4593 没有该分支，本次不纳入绘图样本。

SED／lag 拼版在页顶集中给出两类纵坐标的定义和共用图例，页底只放一次波长说明；各面板保留自己的纵轴数值刻度，最底行显示横轴数值刻度。两类面板的绘图区均为约 1.409 × 1.150 英寸。仅重绘该拼版、保留独立单源图和现有 TeX 图注时，运行 `python src/plot_sed_lag_paper.py --all --layout-only`。

| 图 | 绘图入口 | 论文位置 |
| --- | --- | --- |
| FFA 光变关系与宿主/可变 SED | `plot_ffa_paper.py` | Mrk817_a 示例保留正文，其他源每行两组 FFA＋SED，每页最多六源放附录 |
| MICA 光变分解与原生总转移函数 | `plot_mica_decomposition_paper.py` | Mrk817_a 示例放 Section 3.2，其他源逐页放附录 |
| 分解前/短成分 SED 与短延迟谱 | `plot_sed_lag_paper.py` | 全部源放 Results，每页最多十源，每行两组 SED＋lag |

输出位于 `DRMH0/fig/paper/{ffa,mica,sed_lag}/`，包括单源 PDF/PNG、分页图和数据来源 JSON。分页保留矢量绘图元素。`DRMH0/fig/paper/manifest.json` 汇总全样本路径与正文/附录分配。

统一入口还会生成 `ffa_example.tex`、`mica_example.tex`、`sed_lag_figures.tex` 和 `appendix_figures.tex`，正文通过 `\input` 引入。修改固定图注应同步修改 `render_paper_figures.py`，以免下次生成时覆盖。只更新已有图的插入文件可用：

```bash
python src/render_paper_figures.py --assemble-only --stage mica_gamma2_physw_relaxed
```

## 单源 Fig. 3：SED 与 lag spectrum

`plot_sed_lag_paper.py` 从已经保存的 FFA 和 MICA 结果生成单源双面板图：左侧比较分解前与短延迟成分的可变 SED，右侧展示短延迟成分的 lag spectrum。绘图不会启动 FFA、MICA 或新的拟合。

## 调用

在项目根目录、已激活 `astro` 环境时运行：

```bash
python src/plot_sed_lag_paper.py --source Mrk817_a --stage mica_gamma2_physw_relaxed
```

默认生成 `DRMH0/fig/previews/Fig3_Mrk817_a_sed_lag.pdf`、同名 PNG 和 JSON。JSON 保存输入路径、结果分支、缩放常数、绘图数据及误差定义。

可指定源目录、输出位置和标题：

```bash
python src/plot_sed_lag_paper.py \
  --workspace /home/leonhard/sci001 \
  --source Mrk817_a \
  --stage mica_gamma2_physw_relaxed \
  --output DRMH0/fig/Fig3_Mrk817_a.pdf \
  --title 'Mrk 817 — segment a'
```

也可以在 Python 中导入 `plot(source, stage, output=None, title=None)`，返回 PDF、PNG、JSON 路径字典。脚本保留独立入口，与 `src` 中其他论文绘图脚本一致。

## 输入与定义

读取 `<source>/config/source_config.yaml` 指定的两成分 Gamma 分支，短成分路径为 `<output_root>/2comp/result/comp0`。

松先验分支的现有运行方案只重跑需要调整的波段，其余波段通过软链接复用原 `phys12_width12` 运行；这不是绘图时的回退。当前十个具有该分支的源共 50 个响应波段，其中 27 个重跑、23 个按原配置复用。MICA 单源 JSON 的 `native_data_directory` 记录分支路径，`resolved_native_data_directory` 和 `reuses_existing_run` 明确记录真实路径与复用情况。短成分 SED 与 lag 均读取松先验分支重新汇集的 `result/comp0` 产品。NGC4593 当前没有松先验分支，不能自动标为松先验结果。

| 内容 | 输入文件 |
| --- | --- |
| 分解前 SED | `<source>/<fflux.out>/analysis/<source>_total_6band_flux_posterior.txt` 的 `R_band` 后验 |
| 短成分 SED | `<comp0>/fflux/analysis/<source>_comp0_6band_flux_posterior.txt` 的 `R_band` 后验 |
| 短成分 lag | `<comp0>/lag/<source>_lambda_tau_points.csv` |
| lag 拟合 | `<comp0>/lag/<source>_lambda_tau_fit_summary.csv` 中 `fit_model=4/3` |
| lag 拟合区间 | `<comp0>/lag/tau0_posterior.txt` 中 `tau0_b4_3` |

SED 的原始量为 `lambda_R_lambda`；`R_lambda` 是 FFA 的可变幅度／斜率。绘图从现有 `R_band` 后验计算其 p16、p50、p84；部分旧 `sed_points.csv` 保存的是 `lambda_delta_flux`，不能作为 RMS SED 混用。每条 SED 分别取 W2 的原始中位数作为固定常数 `C_W2`，再将所有波段的 p16、p50、p84 同除以该常数。因此两条 SED 的 W2 中心均为 1，所有点的原有相对误差保持不变，W2 也保留误差棒。这里没有逐后验样本除以自身 W2，也没有额外叠加 FFA 截距或 SED 谱指数的不确定度。此缩放仅用于展示；原始结果文件及距离计算输入不变。

左侧黑色虚线是经过共同 W2 中心的固定 SSD 参考形状 `(lambda/lambda_W2)^(-4/3)`，不是对任一组 SED 数据的拟合。右侧使用观测者系波长与相对 W2 的短成分 Gamma 质心延迟；黑色虚线及灰带来自已保存的 `beta=4/3` 拟合及其后验。图中不混入总转移函数质心或其他观测时段的 ICCF 结果。

## 已确定的样式

单源一行两列，7.2 × 3.3 英寸；衬线字体、向内刻度、细边框。分解前为灰色空心方点，短成分为 Guo et al. (2022) 文中使用的深蓝色 `#00008B` 实心圆点及误差棒。SED 使用对数坐标，lag 使用线性坐标；坐标范围包含误差棒。使用本地 Matplotlib 输出矢量 PDF 和 300 dpi PNG。

## MICA 参数 corner

主流程的 `all`（含 `--stage`）在 `collect` 后自动绘制 corner。只使用已有后验绘图：

```bash
python main.py NGC6814 corner --stage mica_gaussian2_free
python main.py Mrk142 corner --stage mica_gamma2_free100_host
```

每个响应波段的 `<source>_<band>_<n>comp_<model>_corner.pdf` 和同名 300 dpi PNG 保存在该 case 的 `result/corner/`，不写 JSON sidecar。PDF 中的散点、密度、轮廓和文字均保留为矢量，避免 Matplotlib 混合栅格后端为多个面板分配全画布缓冲；PNG 仍为真实 300 dpi 输出。两种格式都先写同目录临时文件，完整导出后才原子替换正式文件，中断不会留下半写文件。单独预览一个波段可调用 `python src/plot_mica_corner.py NGC6814 --stage mica_gaussian2_free --band U`。

双成分 corner 为八维原始采样坐标：`(ln sigmad, ln taud, ln A0, tau0, ln w0, ln A1, tau1, ln w1)`。前两项是 MICA 原生 DRW 振幅参数和时标的自然对数，振幅与宽度也直接读取原始自然对数列，不先指数变换。固定系统误差不画。当前运行的 `tau` 是观测者系质心（天）；历史 `component_center` 列保留原始 center/onset，轴标使用 `c`，不会冒充质心。

读取全部等权 `posterior_sample1d.txt_N`，DRW 与分量始终来自同一行。双 Gamma 强制按每条样本的物理质心升序排列，整组交换振幅、位置和宽度；历史 center 参数用其对应质心判断顺序，但仍绘制原始 center。其他模型沿用 case 的成分排序规则。不截断后验、不移除重复样本、不套用原始采样的 `weights.txt`。

样式来自项目内的 `src/image-creator.mplstyle`，corner 采用 STIX、黑色实线阶梯直方图、灰色区间、橙色中位线、原始散点、浅灰密度及黑色实线等高线。默认一维 25 格、不平滑；灰色边缘区间和顶部摘要始终直接使用原始后验的 15.85/50/84.15 分位数。二维默认每轴 40 格，Gaussian 平滑标准差 1 格；每轴实际带宽上限为原始中央 68.3% 区间宽度的 1/4，避免长尾把平滑尺度拉得过大。只平滑二维密度，不移动样本、不平滑一维分布。

二维等高线在平滑后重新计算，目标为包含 68.3% 和 95.4% 概率的估计区域，不解释为一维 sigma；有限网格与并列值仍会造成离散误差。全部原始散点都在密度上方保留，坐标范围包含全部样本。读取运行保存的参数先验边界，在实际硬边界反射平滑，在仅由样本极值决定的边缘使用零延拓；密度不画到先验外，也不靠强制边界密度为零来制造内部峰。DRW 参数文件中的 `[0,0]` 占位值不当作有效边界。重新排序的成分使用各原生成分边界的并集；已知的质心顺序还约束对应二维密度，历史 Gamma center 不冒充有序质心。

少于 100 个样本或不同联合取值时，下三角仅展示散点；恒定参数涉及的面板也只画散点。若长尾使某个参数中央 68.3% 区间不足三个二维网格宽，该参数涉及的面板也省略密度与等高线，以免把未分辨的主体画成貌似精确的轮廓。全部点和一维分布照常保留。这些规则是显示分辨率选择，不是收敛阈值，也不把不同取值数视为有效独立样本量。绘图函数返回样本数、实际带宽、分位数等信息，但不生成 JSON，不在图内添加方法说明。

可通过独立脚本设置 `--bins`（二维网格）、`--hist-bins`（一维网格）和 `--smooth`（二维平滑标准差，单位为格）；主流程自动使用上述默认值。敏感性对照可用 `--bins 20 --smooth 0`，默认命名下会替换同名图，因此需要保留对照时先另存现有图。分箱或适度平滑改变就消失的小岛不作为确定多峰的证据；显示稳健性不替代采样收敛检查。
