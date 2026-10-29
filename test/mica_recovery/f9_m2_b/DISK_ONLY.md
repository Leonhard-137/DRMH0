# F9 M2/B 纯盘对照

输入位于 `cases/disk_only/input/`，配置位于该目录同级的 `config/source_config.yaml`。
使用原模拟保存的盘变化和噪声，目标光变为 `1 + disk_variation + noise`。
W2 输入逐字节复制，M2/B 的采样时刻和逐点误差保持一致，慢成分设为零。
无需重新生成随机驱动或盘核。

恢复配置与原四组模拟一致：双 Gamma、每个组分质心先验 `[-10,2000]` 天、
宽度先验 `[0.01,1200]` 天、每次 2 MPI ranks、每 rank 4 particles。
M2、B 依次运行，共 2 次拟合；原八次模拟的批次与 manifest 保持独立。

在 astro 环境、项目根目录运行：

```bash
python test/mica_recovery/f9_m2_b/run_disk_only.py
```

结果写入 `cases/disk_only/runs/mica/2comp/`，日志写入 `cases/disk_only/runs/mica/_logs/`。
本次仅准备输入与配置，等待用户自行启动。

纯盘的总相对质心基准为 M2 `0.127983` 天、B `1.179015` 天；完整数值及来源见
`cases/disk_only/truth.json`。该基准来自绝对盘核与 W2 核的质心差。
两条 Gamma 可以共同表示同一个盘响应，不能给每个拟合组分强行分配独立物理真值。
对照时既检查总响应，也检查短分量和额外宽分量如何分配信号。
