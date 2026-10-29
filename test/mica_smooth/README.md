# Mrk 817 高斯平滑实验

代码与四组输入已生成；运行状态见 `monitoring/status.json`。

## 数据和实验矩阵

- `data/raw/Mrk817_a.csv`：来自项目 `Mrk817_a/raw/` 的原始 CSV 留档。
- `data/original/`：来自 `Mrk817_a/fflux/host_sub/` 的六条现有 MICA 输入，已完成消光和宿主扣除。
- 直接使用这些已放入的数据，跳过 MICA 之前的 FFA；MICA 之后的 comp0 FFA 与 SED 属于默认必需步骤。
- W2（UVW2）作为驱动；UVM2、UVW1、U、B、V 为五个响应波段。
- 使用约 416–420 天的完整观测窗口，保留各波段自己的观测时刻。

| case | sigma（观测者系天） | UVW2 | 五个响应波段 |
| --- | ---: | --- | --- |
| `responses_s20` | 20 | 原样 | 高斯平滑 |
| `responses_s40` | 40 | 原样 | 高斯平滑 |
| `all_s20` | 20 | 高斯平滑 | 高斯平滑 |
| `all_s40` | 40 | 高斯平滑 | 高斯平滑 |

每个 case 分别拟合双高斯和双 Gamma，共 8 个分支、40 个波段拟合。
使用原 `Mrk817_a` 的 `mica_gaussian2_free` / `mica_gamma2_free` 配置，
两分量的质心先验均为 0–100 天。保留原采样设置：Gaussian 为 10000 saves、
ptol=0.1；Gamma 为 30000 saves、ptol=0.01。Gamma 的后验按一阶矩排序，
Gaussian 保留原生排序。调度器最多同时运行 3 个 case；每个 case 的 5 个响应波段并行，每波段使用 2 个 MPI rank。各 case 先后拟合双高斯、双 Gamma，再执行完整后处理；一个 case 全部完成后，等待中的 case 立即补上。耗内存的后处理在 case 之间串行。

## 平滑算法与误差

`prepare.py` 在原观测时刻计算归一化高斯加权平均：

    W_ij = exp[-(t_i - t_j)^2 / (2 sigma^2)] / sum_j exp[-(t_i - t_j)^2 / (2 sigma^2)]
    F_smooth = W @ F_original

时间距离按天计算，无误差倒数加权。边界仅对已有点重新归一化，不外推，
不填充。采样较密的时段在这一离散运行平均中占较大权重。
输入 MICA 的是平滑曲线本身；此次操作保留慢变化、压低快变化。
只平滑响应也会改变其相对于原始驱动的有效核，因此结果用于比较平滑操作的影响。

`experiment.yaml` 中 `error_mode: original` 默认保留原观测误差列，作为初步
对照的明确约定。另保存独立测量误差经平滑传播后的完整协方差
`C = W @ diag(error^2) @ W.T` 及其边际误差。
现有 MICA 只读取三列数据，未使用这些非对角项；默认误差约定也不等同于
完整相关噪声似然，后验区间须按这一近似理解。
`error_mode: propagated` 可显式改用 sqrt(diag(C))，但同样忽略相关性，
在相邻平滑点高度相关时容易高估信息量。改变约定应生成新的 case 目录。

## 运行入口

从 `/home/leonhard/sci001` 运行，使用现有科研 Python：

```bash
# 生成输入、配置、协方差和光变对照图；不启动拟合。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/prepare.py

# 查看四组 × 两种模型的实际命令；不写结果、不启动拟合。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/run.py all --dry-run

# 单独运行一组；all包含MICA、分量重建、comp0 FFA/SED和时延分析。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/run.py all --case responses_s20 --model gaussian

# 运行全部8个分支：3个case并行，case内5个响应波段并行。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/orchestrate.py run

# 查看调度器状态。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/orchestrate.py status

# 串行运行全部分支的备用入口。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/run.py all

# 已有MICA结果时，从collect开始补齐分量重建、comp0 FFA/SED和时延分析。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/run.py post --case all_s20 --model gamma

# 读取已有结果，与未平滑的原817分支比较。
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/compare.py
```

`--case` 和 `--model` 支持空格分隔的多选；默认均为 `all`。
准备程序重复运行时只复用一致的输入；发现已有 case 内容不同会停止，
可通过 `prepare.py --output-dir NEW_DIR` 保留另一套数据，随后用
`run.py --cases-dir NEW_DIR` 指向同一目录。不要在已有采样中替换其输入。

## 主流程与图件

`run.py all` 顺序调用：

    mica → collect → corner → transfer → 完整后验分量重建 → comp0 FFA/SED → lagfit

`run.py post` 从 `collect` 开始执行同一路线，包含 comp0 FFA/SED；
`run.py sed` 单独补齐完整后验分量重建和 comp0 FFA/SED。
复用主项目脚本，MICA 的启动、续跑和已完成跳过由现有 `run_mica.py` 处理。
当前 `main.py` 的 `--stage` 允许列表缺少 `collect`，因此这个步骤直接调用
`src/collect_mica_results.py CASE STAGE`，其余步骤调用主程序。

每个 case 的目录结构：

```text
config/source_config.yaml     # 冻结的两种MICA配置
config/{model}_sed.yaml       # 分量重建与FFA/SED使用的配置别名
prepared/                     # 六条本case实际输入
noise/                        # 六个协方差/边际误差npz文件
manifest.json                 # 平滑方法、输入来源和误差约定
input_summary.csv             # 点数、时间范围、原始/处理后RMS
figures/lightcurves.png/.pdf   # 六波段原始和平滑光变对照
logs/                         # 各model/步骤的追加日志
runs/mica/{gaussian,gamma}/free/2comp/
  result/corner/              # 五个响应波段的corner图，PNG+PDF
  result/transfer_functions/  # 逐波段comp0/total及五波段总览，PNG+PDF
  result/branches/full/decomposition/  # 保留分量排序的完整后验重建
  result/comp0/fflux/analysis/ # comp0变量SED图、SED拟合及FFA后验数据
  result/comp0/lag/           # lag spectrum及其拟合表
```

`transfer` 同时生成后验区间和中位参数核两套图。corner 在分支之间串行绘制。
`compare.py` 汇总各 case 和原始817结果中的质心、MICA宽度参数、负时延面积比例，
输出 CSV 和分模型对照图。Gamma 的宽度参数是 scale，不与 Gaussian 的 sigma
冒充同一个 RMS 宽度。负时延面积由逐个后验样本计算，total 按每个样本的分量
面积加权；不存在的结果明确列在缺失表中。

## MICA 后的 comp0 FFA 与 SED

`all` 和 `post` 默认生成主程序定义的 comp0 变量 SED，无需附加参数。
已有 MICA 结果时，可单独补跑这一部分：

```bash
/home/leonhard/miniforge3/envs/mica2/bin/python -B test/mica_smooth/run.py sed --case all_s20 --model gamma
```

此路线使用完整后验范围重建分量，再调用现有 FFA/SED 程序，不切选后验；
重建使用固定种子和3000次后验抽取。Gamma 保留按质心排序的身份，
Gaussian 保留原生排序，不直接把未逐样本重排的 Gamma 原生 comp0 光变送给 FFA。
驱动使用本 case 的 `prepared/Mrk817_a_UVW2.txt`，响应使用各自重建的 comp0 光变。
输入已完成消光和宿主扣除，case配置的额外 E(B-V) 为0，后FFA步骤不再扣宿主。

FFA 执行 `prep → dered → fit → summ → fvg → sedfit`，其中 `dered` 使用零额外消光。
实际 SED 图和数据位于各分支 `result/comp0/fflux/analysis/`，包括
`Mrk817_a_comp0_sed_models.png/.pdf`、`Mrk817_a_comp0_sed_fit_summary.csv`、
`Mrk817_a_comp0_sed_points.csv` 和相应后验文件。
已经完整的SED可复用；若FFA留下未完成的样本，脚本会报告路径并停止该步骤，
由现有FFA工具恢复，避免重新fit覆盖。`all/post` 完成分量SED后继续 `lagfit`；
`sed` 在分量SED完成后结束。这些入口均不运行距离或H0。

## 验证

```bash
/home/leonhard/miniforge3/envs/mica2/bin/python -B -m unittest discover -s test/mica_smooth -p 'test_*.py' -v
```

测试覆盖不等间隔权重、恒定信号保持、理想频率衰减、协方差传播、W2处理范围、
分支配置、重复准备、默认包含后FFA/SED的主流程顺序、Gamma重建身份、
替代case目录，以及dry-run不执行命令、不改文件。测试不启动MICA/FFA采样器。
