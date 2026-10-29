# 隐藏驱动参考系的补充核对照

只展示 M2/B 的 response_within 和 response_beyond，不运行新的拟合。

三列分别比较盘部分、慢部分、总核。真实绝对盘核直接读取实际生成器保存的 cell mass；真实绝对慢核为实际慢核乘注入 gain 后与 W2 绝对盘核卷积。真实总核为两者之和。

恢复曲线来自已有完整加权后验 mean_small/mean_large/mean_total；先在 0.02 天网格上线性插值，再与 W2 绝对盘 cell mass 进行完整线性卷积。平均与线性卷积可交换，因此无需重新抽样。没有对点位后验区间作卷积；本图只展示均值，不展示不确定度。

小/大质心标签不自动等于盘/慢物理分量。左右对照检验其对应程度，不能预设分量识别成功。核增益沿用输入通量单位，无额外面积归一化。数值误差、原缓存来源和面积核验见 validation.json。输出 NPZ 是稀疏显示网格，积分验证采用完整 0.02 天网格。

- [M2 PNG](F9_M2_forward_truth_comparison.png)
- [M2 PDF](F9_M2_forward_truth_comparison.pdf)
- [B PNG](F9_B_forward_truth_comparison.png)
- [B PDF](F9_B_forward_truth_comparison.pdf)
