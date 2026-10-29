# F9 全波段慢信号恢复测试配置

本目录接续 `../f9_m2_b` 的既有模拟，仅补充 W1、U、V 波段。
W2 仍是观测参考波段；M2、B 已完成的 10 次拟合直接复用，新增 15 次拟合，
最终得到 5 个条件 × 5 个响应波段，共 25 组结果。
配置与输入准备不启动 MICA；拟合需执行下方运行命令。

| 条件 | 注入内容 | 本次补跑 |
|---|---|---|
| `disk_only` | 纯盘及测量噪声 | W1、U、V |
| `response_within` | 盘＋真实慢响应，Gamma 质心 60 天、宽度 30 天 | W1、U、V |
| `random_within` | 盘＋对应低频 PSD 的独立慢变化 | W1、U、V |
| `response_beyond` | 盘＋真实慢响应，Gamma 质心 800 天、宽度 400 天 | W1、U、V |
| `random_beyond` | 盘＋对应低频 PSD 的极慢独立变化 | W1、U、V |

## 固定的模拟和拟合设定

采用 F9_a 各波段实际时间戳和逐点相对误差，保留原实验的隐藏 DRW、
两档独立慢信号、时间零点和 W2 输入。新增波段使用独立噪声种子分支，
各条件共用同一波段的盘变化和噪声。每个混合条件在对应观测时刻的
慢变化/盘变化 RMS 比仍为 0.3；纯盘条件该比值为 0。

每个目标波段分别拟合 W2→目标的双 Gamma 转移函数。两组分均使用质心
`[-10, 2000]` 天、log-uniform 宽度 `[0.01, 1200]` 天的先验，
其余推断设置继承同条件的旧 M2/B 配置：`component_order: moment1`、
`max_num_saves: 10000`、`num_particles: 4`、`ptol: 0.1`。
本次仅扩展波段，未改为真实 F9 宽先验拟合的那套边界。

每个拟合使用 2 MPI ranks，每 rank 4 particles；每个条件依次运行 W1、U、V，
最多同时运行 2 个条件，因此最多 4 个 MICA ranks。运行入口调用项目
`src/run_mica.py`，请勿在命令外再套 `mpiexec`。

标准转移函数导出范围仍为 `[-100, 500]` 天、12001 点。这是绘图窗口，
不限制上述拟合先验；它不能展示长分量的全部支撑。核矩和面积应由完整
核或参数计算，不能从该截断绘图窗口积分。

## 检查与运行

在项目根目录、自己的持久终端中执行。先检查已准备的配置和输入：

```bash
/home/leonhard/miniforge3/envs/astro/bin/python -B test/mica_recovery/f9_all_bands/run_all_bands.py --validate-only
```

验证模式只读文件：检查新输入、旧 10 组必要输出、先验和调度设置，
不会创建运行目录、日志或状态，也不会启动拟合。

确认验证通过后，运行全部 15 次补充拟合：

```bash
/home/leonhard/miniforge3/envs/astro/bin/python -u test/mica_recovery/f9_all_bands/run_all_bands.py
```

同一批次已有状态、日志或新结果时，运行入口拒绝覆盖，也不会自动重试或续跑。
正常终端中断会停止该批次已启动的条件调度器及其 MPI 子进程。

## 配置、进度与输出

- `batch_config.yaml`：波段、条件及并行上限。
- `cases/<condition>/config/source_config.yaml`：各条件的 MICA 配置；实际待跑响应仅为 W1/U/V。
- `cases/<condition>/input/`：新增目标及共用 W2 输入。
- `outputs/truth.json`：注入真值与构造记录；真值采用本次盘核相对 W2 的定义。
- `manifest.json`：全部 25 组结果路径及 `reused` 标记。
- `run_status.json`：启动后的条件进度、退出码及缺失输出。
- `logs/<condition>.log`：条件调度日志；MPI 波段日志位于该条件的 `runs/mica/_logs/`。
- 新结果：`cases/<condition>/runs/mica/2comp/run_W2_to_<band>_2comp_gamma/`。
- 旧 M2/B 结果：仍位于 `../f9_m2_b/cases/<condition>/runs/mica/2comp/`，不复制、不重跑。

本入口不调用旧版只针对 M2/B 的汇总脚本；拟合结束以状态和实际后验文件为准。

## 科学解释范围

盘核沿用单位面积归一化、各带平均通量为 1 的构造。本次可以比较各波段盘延迟、
变化幅度及光变恢复能力，但未额外注入物理 variable SED 或 U 波段超额，
不能直接作为真实 SED/U 超额验收。仍然只有一个配对随机实现，不能据此估计
覆盖率或检出率。短质心组分是盘候选；两个拟合 Gamma 也可能共同分配盘变化。
独立慢信号不存在注入的总转移函数，不能为它定义真总响应核或真总 lag。
