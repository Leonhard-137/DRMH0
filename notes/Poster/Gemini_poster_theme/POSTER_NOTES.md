# 一分钟中文海报

题目：活动星系核测量宇宙学距离的潜力

A1 横版（841 × 594 mm），Gemini 原有蓝白配色，保留用户选定的 IHEP 标识。
正文与图外说明改为中文；图内文字、配色、线型与面板保持原样。作者姓名采用论文中可核实的英文拼写。
结构已由用户确认：测距原理 → 本工作 → H0 结果 → 宇宙学潜力。
研究目标是测量距离、约束哈勃常数；MICA 分解及其检验是测距方法的一环。

## 原图与结果来源

直接嵌入以下两个既有 PDF，不运行绘图脚本，不裁切面板，不修改原图：

- 主图：../../../DRMH0/fig/paper/hubble/joint_hubble_shared_sed_no_f0.pdf
- 辅助图：../../../DRMH0/fig/paper/sed_lag/Fig3_Mrk817_a_sed_lag.pdf

正文依据：../../../DRMH0/sample701.tex 及其 Analysis、Results、Discussion。
联合结果依据：../../../DRMH0/tables/relaxed/joint_result.csv 和主图 manifest。
所用结果为 mica_gamma2_physw_relaxed：10 个 AGN，每个源一个距离约束。
H0 = 89.6 (+8.3, −7.0) km/s/Mpc，χ² = 49.6047，dof = 9。
结果依赖共享标准盘 SED、固定倾角 45°、零额外散布与无本动速度项；不同拟合阶段间的不确定度未完整传播。
短延迟成分不因质心更早就被确认为纯盘信号。

上一版新绘的 U 超额图、重绘的 Mrk 817 图及其脚本均不被本版引用；保留这些旧文件，不做额外清理。
论文正文和原始图文件未修改。

## 编译

在本目录运行：

    latexmk -xelatex -interaction=nonstopmode -halt-on-error poster.tex
    pdftoppm -png -singlefile -scale-to 2400 poster.pdf poster-preview

也可运行 make。无需运行 build_poster_figures.py。

当前使用系统已有的 Windows 微软雅黑字体，路径为 /mnt/c/Windows/Fonts/msyh.ttc 与 msyhbd.ttc。
正文使用 XeTeX 原生中文断行；数学及科学图继续保留原有字体。

## 约一分钟中文讲述顺序

我的工作探索活动星系核测量宇宙学距离的潜力。连续谱反响映射利用不同波段的时延测量吸积盘尺度，再结合流量估计距离，进而约束哈勃常数。

我们分析了十个 AGN 的六波段光变，用 MICA 提取不同响应成分，并通过能谱和延迟谱检验短延迟成分，再将其用于测距。

右侧是主要结果：十个源的距离—红移关系给出约 90 的哈勃常数条件估计，但当前拟合仍存在明显的模型失配。

因此，我们探索的是一条基于 AGN 光变的测距路径。未来需要控制系统误差，扩大样本并拓展到更高红移，进一步发挥它的宇宙学测量潜力。
