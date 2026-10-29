# NGC2617_a 联合与逐波段 MICA 结果图

本目录按 2026-09-09 的看图请求生成，来源是
`NGC2617_a/runs/mica/gamma/broad_np4/2comp/` 下的六个已完成任务。
输入、先验和采样设置见上一级 `README.md`。

## 图件

- `native/native_light_curves_all.pdf`：6 页原生光变图，依次为联合、M2、W1、U、B、V。
- `native/joint/native_full.pdf`：联合任务的完整 MICA 原生图，保留光变曲线、残差和传递函数等原生面板。
- `native/pair_{band}/native_full.pdf`：每组逐波段任务的完整原生图。
- `parameter_histograms.pdf`：7 页参数图；前 5 页各波段的 6 个响应参数，第 6 页 DRW 参数，第 7 页小质心局部图。
- `parameters_overview.png` / `.pdf`：五个波段的 30 个响应参数总览。
- `parameters_{band}.png`：每个波段的响应参数大图。
- `drw_parameters.png`：2 个自由 DRW 参数；重复显示同一联合后验，与五组配对分别比较。
- `centroid_detail.png`：小质心参数在 -2 至 4 天的局部图，标注窗口内概率，保留完整后验归一化。
- `parameters_native_labels.png` / `.pdf`：按原始 MICA 分量编号显示的同一批后验，保留标签交换。

## 参数图的定义

蓝色实线为联合，橙色虚线为逐波段。使用 `sample1d.txt_2` 的全部 10,000 行，
将 `weights.txt` 最后一列除以其总和，得到每行后验权重。
权重文件中的 likelihood 与 `sample_info1d.txt_2` 逐行核对一致。
未裁剪后验、未作额外重加权、未作平滑。

每个响应波段按每条样本的质心大小，交换完整的 `(ln A, tau, ln w)` 三元组。
小／大质心只是排序标签，不预先指定盘或 BLR 身份。
本批 `centroid` 列已经是质心，不能再加 `2w`。
宽度 `w` 是 Gamma 尺度参数，振幅 `A` 使用 MICA 内部归一化坐标。
宽度和振幅直方图采用 `log10` 横坐标，DRW 图采用原生自然对数坐标。
纵轴是共同分箱内的后验概率，局部图也不重新归一化。

联合任务共有 38 个参数槽，其中 30 个响应参数和 2 个 DRW 参数自由变化；
连续谱及五个响应的额外系统误差参数固定为 0，因而不画伪造的连续分布。
原始观测误差仍然参与拟合。

## 原生图及核查

`../render_native.py` 在独立副本中通过原生 C 函数
`output_reconstruction_parallel` 读取保存的等权后验文件，再调用安装版
`pymica.utility.plot_results` 绘图。没有调用 sampler 或 `post_run`，
没有重抽后验。原生光变区裁图只提取完整原生图中的光变及残差面板。
各副本的 `render_manifest.json` 记录原生库、后验行数和正式文件未改变的核查。

参数图使用加权原始样本，原生重建使用 MICA 已保存的等权重采样后验；
两者是同一拟合的两种保存产物，但有限重采样会有差别。
Kish ESS：联合约 739；M2/W1/U/B/V 分别约 1440/1318/984/1576/1555。
这些数值衡量权重集中程度，没有校正序列相关性，不能单独证明收敛。

`parameter_summary.csv` 给出绘图坐标中的 16%、50%、84% 分位数；
`histogram_bins.csv` 保存所有完整范围直方图的分箱和概率；
`posterior_metadata.json` 保存路径、样本数、ESS 与标签交换概率。

重新生成参数图：

```bash
/home/leonhard/miniforge3/envs/astro/bin/python -B test/mica_joint/plot_parameter_histograms.py
```

重新生成原生图：

```bash
/home/leonhard/miniforge3/envs/astro/bin/python -B test/mica_joint/render_native.py
```
