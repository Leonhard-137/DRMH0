# NGC2617_a：全部六波段联合 / 逐波段 MICA 对照

本测试只运行 MICA 拟合。W2 为驱动，M2、W1、U、B、V 为响应，
两种方式读取同一批 `NGC2617_a/fflux/host_sub` 文件，合计 257 个观测点。
这里的“单波段”指每次拟合 W2 与一个响应波段。

配置为 `NGC2617_a/config/source_config.yaml` 中的 `mica_gamma2_broad`：
双 Gamma，`type_lag_prior=4`，两个分量的质心先验均为 [-10, 100] 天，
宽度采用该波段与 W2 配对时的原生默认先验。联合模式通过参数先验文件
保留每个波段的这个范围，避免增加波段时自动改变宽度先验。
不对不同波段的响应参数施加相等关系或波长关系；联合模式只在一个数据集内
共同拟合驱动及全部响应，包含原生的响应间协方差。

NGC2617_a 原有物理窄先验结果不能用作本测试的逐波段基线，所以另跑同配置
的五组 W2–响应配对。正式设置为每任务 2 个 MPI 进程、每进程 4 个粒子、
10,000 个保存样本、`ptol=0.1`。各任务顺序运行，避免它们互相争抢 CPU。
达到保存数量只说明采样任务结束，不能代替收敛或有效样本量检查。

在项目根目录运行（使用已验证的 astro 环境，启动器会自行调用 MPI）：

```bash
# 独立联合入口
/home/leonhard/miniforge3/envs/astro/bin/python -B src/run_mica_joint.py NGC2617_a --stage mica_gamma2_broad

# 已整合到原脚本的等效入口
/home/leonhard/miniforge3/envs/astro/bin/python -B src/run_mica.py NGC2617_a --stage mica_gamma2_broad --joint

# 五个逐波段对照；同一配置，仍然只拟合
/home/leonhard/miniforge3/envs/astro/bin/python -B src/run_mica.py NGC2617_a --stage mica_gamma2_broad --fit-only
```

默认 `run` 会检查并跳过已完成的同配置任务，或恢复有效断点。
可在源名后加 `resume` 显式续跑；保持输入、先验、MPI 数量和环境不变，
仅增加 `max_num_saves` 累计目标。`fresh` 会另行归档已有任务，通常不需要。

`bash test/mica_joint/run_ngc2617_a.sh` 会依次执行联合及五组对照。
2026-09-09 正式批次已通过 tmux 会话 `mica_joint_ngc2617_a_20260909` 启动，
启动日志为 `test/mica_joint/production_launcher.log`；各任务详细日志在
`NGC2617_a/runs/mica/gamma/broad_np4/_logs/`。已运行时不要重复启动同一批次。

输出位于 `NGC2617_a/runs/mica/gamma/broad_np4/2comp/`：

- 联合：`joint_W2_to_M2_W1_U_B_V_2comp_gamma/`。
- 对照：`run_W2_to_{band}_2comp_gamma/`。
- 每个目录保存输入副本、实际参数先验、样本、后验、证据、断点及
  `run_mica_state.json`（包含实际采样阶段耗时）。联合样本 38 列，对照 10 列。

`fit_only` 保留连续谱拟合及响应采样所需的原生后验处理；连续谱阶段仍有
MICA 自带的中间产物。响应阶段不生成 `pall.txt_2` 或 `pline` 分解，
不进入 Python 绘图、分支分解、SED、lag 或距离/H0 流程。

代码验证用 `smoke_20260909/`，读取相同六波段原始输入，使用 1 个粒子和
50 个保存样本起跑，再续跑到 100。该目录仅用于流程及断点验证，不能作为科学结果。
离线回归测试：

```bash
/home/leonhard/miniforge3/envs/astro/bin/python -B -m unittest discover -s test -p 'test_run_mica*.py' -v
```

2026-09-09 六项正式拟合已完成。用户随后授权展示原生光变曲线及参数直方图，
对应图片、PDF、绘图方法和参数定义见 [结果图说明](figures_20260909/README.md)。
