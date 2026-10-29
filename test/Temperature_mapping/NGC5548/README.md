# NGC 5548 temperature mapping

主结果使用 `/home/leonhard/sci001/NGC5548/raw/NGC5548.csv` 的19个UV/光学波段，4,203点。原始文件、817输出和MICA配置均不修改。结果解释先读 `RESULTS.md`。

这是NK22方法级复现；数据版本与未知作者配置使它不是精确数值复制。公开5900点TempMap包有混合时间系统问题，只作诊断，不能当作无问题的原文输入。

## 重跑

在 `/home/leonhard/sci001/test/Temperature_mapping/NGC5548` 运行。脚本只写本目录。使用已有环境，无需安装包。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/home/leonhard/sci001/test/Temperature_mapping/NGC5548/outputs/.mplconfig

/home/leonhard/miniforge3/envs/astro/bin/python -B run_reproduction.py
/home/leonhard/miniforge3/envs/astro/bin/python -B sensitivity_checks.py
/home/leonhard/miniforge3/envs/astro/bin/python -B validate_numerics.py
/home/leonhard/miniforge3/envs/astro/bin/python -B audit_sources.py

# 比温度线性求解慢；267个Swift U点，不做时间降采样。
../outputs/lag_env/bin/python -B fit_residuals.py

/home/leonhard/miniforge3/envs/astro/bin/python -B plot_results.py
/home/leonhard/miniforge3/envs/astro/bin/python -B summarize_results.py
```

`fit_residuals.py` 是可选的2024图16条件残差检验，不是MICA或联合盘–BLR模型。若只需要温度图，可跳过它；图册会跳过第08图，结果说明会明确尚无残差后验。

绘图应用科研图样式的可读性、向量PDF和300dpi输出约定；原论文黑点/红色与青色拟合线、分级红蓝温度图的视觉语言优先。每幅图的解释写在 `outputs/figure_captions.md`，未在图中堆入说明段落。

## 依赖和数据

- 复用上级 `temperature_mapping.py` 的已验证稀疏线性反演、`validate_model.py` 的独立测试、`residual_lags.py` 的条件协方差模型；没有修改这些共享文件。
- 绘图样式入口：`/mnt/c/Users/26330/.codex/skills/image-creator/scripts/plot_style.py`。
- 下载来源：`https://raw.githubusercontent.com/Zstone19/TempMap/main/Examples/NGC5548_Data.npz`，保存为 `reference/NGC5548_TempMap.npz`。重命名没有改变其数据。
- 来源教程原样保存为 `reference/TempMap_tutorial.ipynb`，不执行该第三方notebook。
- 原论文PDF位于上级 `reference/neustadt_kochanek2022.pdf`、`reference/neustadt2024.pdf`。
- 数组均 `allow_pickle=False` 读取。所有主结果带 `local_` 前缀；`public_` 前缀明确是来源异常的诊断结果。
