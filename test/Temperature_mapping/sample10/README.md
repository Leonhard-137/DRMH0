# 十源温度映射测试

样本来源：`/home/leonhard/sci001/DRMH0/tables/sample_observations.csv`。
主结果见 [RESULTS.md](RESULTS.md)，图册见 `outputs/temperature_mapping_atlas.pdf`。

使用现有 astro 环境；不安装依赖，不运行或改写 MICA。不改原始数据，之前的 Mrk817 和 NGC5548 图册保留。

```bash
cd /home/leonhard/sci001/test/Temperature_mapping/sample10
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/home/leonhard/sci001/test/Temperature_mapping/sample10/outputs/.mplconfig
/home/leonhard/miniforge3/envs/astro/bin/python prepare_inputs.py
/home/leonhard/miniforge3/envs/astro/bin/python run_batch.py
/home/leonhard/miniforge3/envs/astro/bin/python summarize_results.py
/home/leonhard/miniforge3/envs/astro/bin/python verify_outputs.py
/home/leonhard/miniforge3/envs/astro/bin/python plot_results.py
```

`run_batch.py --sources Mrk335 NGC6814` 可仅计算指定源，但顶层 summary 此时也只含该子集；完整交付应重新对全样本汇总。
`--reuse` 是显式复用已完成输出，调用者须先核对输入和配置未改变；不会静默把旧参数结果标作新结果。

物理参数固定于 `physical_parameters.json`，不得按图样是否符合预期调整。
50×250 主网格、ξ=1/10/100/1000、7/28/56 天低／高通、Nt=100/500 对照、去 U/u、仅 Swift 六波段均保存。
每源独立输出完整 NPZ、指标 JSON 和数值验证；失败源明确进入 `outputs/batch_status.json`，不会被悄悄删掉。

采用 NK22/AGN STORM 2 论文的离散红蓝温度图、时间—对数半径布局，并继承 image-creator 的 STIX、向内刻度和灰色无帽误差棒。
同源分解图沿用主图同一个 q99 色标；另外提供跨源相同绝对色标的总览，避免自动缩放冒充幅度可比。

此轮是固定参数下的正则化映射与稳定性诊断，不包括温度像素后验、传播速度显著性、MICA 联合拟合或十源残差时延 MCMC。

## 图册生成与中断恢复

绘图逐页保存 PDF/PNG/JPEG，关闭并清理每张图后再绘制下一页；最后用 pypdf 合并已完成的单页 PDF，核对页数后替换总图册，避免长生命周期的 Matplotlib PDF 对象保留全部栅格缓冲区。
`outputs/render_progress.json` 记录已完成页数和最近图页；只有 `status=complete` 且总图册可正常解析时才算绘图完成。
若绘图中断，重跑 `plot_results.py` 即可重新生成图页，不要因此重跑 `run_batch.py`。
