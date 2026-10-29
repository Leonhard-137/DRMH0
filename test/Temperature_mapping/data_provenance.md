# Mrk 817 温度反演输入的数据来源与处理

## 最终输入

`derived/lightcurves.csv` 共 7451 行、17 波段，时间为观测者系
THJD = HJD − 2450000，通量及误差单位均为
10⁻¹⁵ erg s⁻¹ cm⁻² Å⁻¹。`error_original` 保存本次误差扩展前的误差，
`error` 为按温度论文 triplet 方法增加系统误差后的值。
`derived/lightcurves_original_errors.csv` 将拟合误差也设为原始误差，供敏感性分析。

| 数据组 | 波段 | 点数 | 原始时间及通量单位 |
|---|---|---:|---|
| HST/COS | 1180, 1398, 1502, 1739 Å | 660 | 出版表 HJD−2400000；10⁻¹⁵ flux |
| Swift/UVOT | UVW2, UVM2, UVW1, U, B, V | 1917 | 出版表 HJD−2450000；10⁻¹⁵ flux |
| Ground | u, B, g, V, r, i, z | 4874 | 出版表 HJD−2400000；10⁻¹⁴ flux |

各带点数、时间覆盖、采样间隔和误差扩展见 `derived/band_summary.csv`。
完整机器可读来源与验证结果见 `derived/data_metadata.json`。

## 一手来源及时间核对

- HST：[Homayouni et al. 2023, ApJ 948, 85, Table 4](https://doi.org/10.3847/1538-4357/acc45a)。
  原始机器可读表保存在 `reference/apjacc45at4_mrt.txt`。
  同一 campaign 的 [MAST 数据集](https://stdatu.stsci.edu/hlsp/storm2)，
  DOI 10.17909/n734-k698，提供 165 次 COS 观测的校准光谱和光变。
- Swift：[Cackett et al. 2023, ApJ 958, 195, Table 1](https://doi.org/10.3847/1538-4357/acfdac)。
  原始机器可读表保存在 `reference/apjacfdact1_mrt.txt`。
- Ground：[Montano et al. 2026, Table 3](https://doi.org/10.3847/1538-4357/ae6103)。
  原始文件为 `Mrk817_data/ground_2026/apjae6103t3_mrt.txt`。

本地 HST+Swift CSV 的时间列标为 MJD，但其转换历史不明。
将它加 0.5 后，与出版表 HJD−2400000 的差值并不严格为零：
Swift 最大差 0.00234626 天，HST 最大差 0.00425142 天。
因此没有依据标签盲目做常数平移，而是每带按时间排序，与出版表逐点配对，
确认点数、通量和误差一致后，直接采用正式表的 HJD 时间。
Swift 的 flux/error 数值完全相同；HST flux 最大差 4×10⁻⁶（上述通量单位），
与本地浮点量化一致，error 完全相同。脚本遇到匹配失败会报错。
最终 flux/error_original 仍保留用户本地表的数值。

地面表的 HJD−2400000 减 50000 得 THJD，flux 和 error 乘 10。
不能将地面时间当作 MJD 再加或减 0.5 天，也不能将地面通量当作 10⁻¹⁵ 单位。

## 波长

HST/Swift 使用 Cackett et al. Table 2 的观测系波长；地面使用
Montano et al. 2026 Table 4 的观测系中心波长：
u=3540, B=4361, g=4770, V=5448, r=6215, i=7545, z=8700 Å。
温度反演论文正文说明最红端为 8897 Å，但没有逐一列出地面波段中心波长；
因此本次采用与最终公开地面数据配套且有明确表格来源的 8700 Å。
这一选择应作为对原论文设置的差异记录，不能宣称参数完全相同。

## Triplet 系统误差

按 [Neustadt et al. 温度论文 §2.1](https://arxiv.org/html/2310.01497#S2.SS1)：
每三个相邻点拟合一条直线，3 个数据减 2 个拟合参数，剩 1 个自由度。
给该三点各误差增加同一非负 quadrature floor，使最小加权 χ²=1；
若原 χ²≤1，该三点 floor=0；取所有三点组 floor 的中位数作为全带系统误差。
最终 σ_new=sqrt(σ_original²+floor²)。没有硬编码误差倍数，也没有按拟合结果调参。

实现用三点加权直线残差的等价解析表达式。令
a=(t₁−t₂, t₂−t₀, t₀−t₁)，则 a 与常数项和 t 均正交，
最小 χ²=(a·f)²/[Σaᵢ²(σᵢ²+s²)]。
令 χ²=1 可直接求出 s²=max(0,[(a·f)²−Σaᵢ²σᵢ²]/Σaᵢ²)。
这是重新加权拟合的解析结果，采用 ddof=1。

本次得到 HST 1180/1398/1502 的 floor 约为
2.05416 / 1.23089 / 0.922938，误差比值的中位数为
3.56666 / 3.47377 / 3.33595。其余 14 带 floor 均为 0。
这与原文“蓝端 HST 需增误差、其余不变”的结论一致；
1180 的具体倍率略高于原文笼统的 3.3–3.5，没有人为调整去追逐该数字。
所有相邻三点组均保留，原文未规定跨长间隙三点组剔除。
原文的 HST 1709 Å 显然对应数据和图中的 1739 Å 带。

## XRT 对照

`derived/xrt.csv` 含同一 Swift Table 1 中 414 个观测时刻的
0.3–10 keV 总计数率，列为 `thjd,rate,error`。
Table 1 只给 S(0.3–1.5 keV) 和 H(1.5–10 keV)，两者 414 个时间戳完全一一相同。
采用严格相等配对（时间容差 0），总率为 S+H，误差为 sqrt(σS²+σH²)，
假设两个不重叠能段误差独立；出版表没有提供其误差协方差。
时间覆盖 THJD 9188.412–9692.133，单位为 count s⁻¹。

## 与 2024 年输入的差异和保留边界

- 2026 最终公开的地面数据使总数成为 7451；2024 温度论文报告 7380。
  这是一组有来源的更新数据复现，不是原作者私有中间版的逐行复刻。
- 未额外做银河系消光修正、宿主扣除、长期去趋势或跨仪器通量调整。
  地面文件已经是作者发布的交叉定标产品，其 telescope 字段被保留。
- 不同望远镜的同时间地面点保留为不同观测，不擅自合并。
- 准備过程没有把波长或时间转到静止系，反演模型必须只在需要处统一应用红移。
- 光变或温度条纹属于模型约束下的反演结果；数据准备本身不赋予其物理机制标签。

## 重跑

```bash
/home/leonhard/miniforge3/envs/astro/bin/python test/Temperature_mapping/prepare_data.py
```

存在正式源表时复用本地副本；缺失时仅下载两张指定出版表。
