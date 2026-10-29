# Mrk 817 温度波动 mapping 复现

使用当前公开的 HST + Swift + 地面 17 波段数据，独立实现 Neustadt et al. (2024) 的 NK22 线性温度反演。

先看 [RESULTS.md](RESULTS.md)；图集位于 [outputs/Mrk817_reproduction.pdf](outputs/Mrk817_reproduction.pdf)，单图位于 outputs/figures/，数据出处见 [data_provenance.md](data_provenance.md)。

**不是原始 2024 输入的逐点重放：** 本次 7451 点，原文 7380 点；地面数据为 2026 最终公开版。Netzer 逐波段模板未取得；NGC 5548 不在本目标数据范围。温度 mapping 主实验完成，但两组残差后验均未通过本次收敛标准，且注入组延迟未对齐原文；图中已标 PRELIMINARY，详见 RESULTS.md 和 outputs/lag_summary.json。

## 在当前 WSL 环境重跑

以下命令从本目录执行，分别运行，不需启动 MICA。温度反演和残差延迟是两套成本不同的计算。

~~~bash
cd /home/leonhard/sci001/test/Temperature_mapping
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

/home/leonhard/miniforge3/envs/astro/bin/python prepare_data.py
/home/leonhard/miniforge3/envs/astro/bin/python temperature_mapping.py
/home/leonhard/miniforge3/envs/astro/bin/python validate_model.py
/home/leonhard/miniforge3/envs/astro/bin/python sensitivity_checks.py
/home/leonhard/miniforge3/envs/astro/bin/python summarize_results.py

# 昂贵的残差延迟拟合。脚本自动使用任务目录内的隔离 JAVELIN 环境。
/home/leonhard/miniforge3/envs/astro/bin/python residual_lags.py --walkers 24 --burn 500 --steps 1000 --max-steps 2000 --workers 2

# 已有有效的逐组缓存时可在上一条命令中增加 --reuse。
# --reuse 只适用于相同输入及先验；更换数据/先验必须使用新输出目录并重新采样。
# 延长采样应检查现有诊断并有意设置新输出，不能把旧缓存误当更长新链。
/home/leonhard/miniforge3/envs/astro/bin/python plot_reproduction.py
/home/leonhard/miniforge3/envs/astro/bin/python qa_outputs.py --final
~~~

脚本会更新本目录 outputs 中相应结果。若要保留不同实验，请先使用不同输出目录或备份相应结果；不要把 --quick 的 25×100 小网格输出混作论文网格结果。

主温度 suite 实测 20.67 秒，不包括数据下载、绘图和 MCMC。重要数值在 outputs/metrics.json，完整模型数组在 outputs/fits.npz。温度条纹是给定盘模型、正则项和误差下的反演结果，不是空间分辨的直接温度观测。
