# F9 单位面积转移函数比较

仅展示 M2/B 的 response_within、response_beyond 四个已完成结果。
沿用预设核与恢复核的正向卷积对照，统一到隐藏驱动参考系。

对上一版的每条后验平均核和预设核，分别除以自身完整计算网格上的积分，使面积为 1。
这是“后验平均核再归一化”，不是“每个样本归一化后再求平均”。不改变样本或其权重，仅改变显示幅度。没有反卷积或新拟合。

盘分量、慢分量各自归一化；总核先保留原增益相加，然后按总面积归一化，因此总核仍保留其原有分量比例。
只画均值，不展示后验区间。归一化积分包含负延迟和完整尾部；局部坐标窗口不参与归一化。
正向卷积和原始缓存误差、归一化前后面积及来源均记在 validation.json。
输出 NPZ 为显示网格，不用于重新积分；面积校验采用完整 0.02 天网格。

- [两页总 PDF](F9_response_unit_area_comparison.pdf)
- [M2 PDF](F9_M2_unit_area_comparison.pdf)
- [B PDF](F9_B_unit_area_comparison.pdf)

![M2](F9_M2_unit_area_comparison.png)

![B](F9_B_unit_area_comparison.png)
