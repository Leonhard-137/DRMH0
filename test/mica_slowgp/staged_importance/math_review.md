# 冻结 proposal 的分层重要性采样：独立数学审查

范围：现有 W2 + 一个响应波段、原有 11 参数 shared-slow 模型；不改变似然、物理先验、三层边界或冻结 gate。这里只给公式与验收要求，未运行科学拟合。`frozen` 指训练结束后完整固定 proposal，评价阶段另抽新样本。

## 1. 原目标、proposal 与逐行权重

层 s 的原先验质量记为 rho_s；[-3,5]、[5,20]、[20,28] 对应 (64,465,432)/961。原条件先验变换 theta=T_s(u)，u 在 (0,1)^d 均匀，d=11。逐坐标 z=logit(u)，故

\[
\pi_z(z)=\prod_j u_j(1-u_j),\qquad
\log\pi_z=-\sum_j\{\operatorname{softplus}(z_j)+\operatorname{softplus}(-z_j)\}.
\]

这已包含 unit-cube 到 z 的 Jacobian；**不能再乘物理先验密度、center 排序因子或另一个 Jacobian**。层内条件先验和原层质量分开处理。

固定 epsilon>0、sum alpha_k=1、alpha_k>=0、nu>0、每个 S_k 正定：

\[
q_s(z)=(1-\epsilon)\sum_k\alpha_k t_\nu(z;m_k,S_k)+\epsilon\pi_z(z),
\quad w_{si}=\frac{L(T_s(u_{si}))\pi_z(z_{si})}{q_s(z_{si})}.
\]

S 是 Student-t 的 **scale matrix**，不是其 covariance；nu>2 时 covariance=nu S/(nu-2)。必须用完整归一化混合密度作分母，不是被抽中的单分量密度。逐行随机抽混合分量再抽 z，才能直接使用下文 iid 方差公式；固定各混合分量的数量会成为另一种分层设计，不能沿用同一 iid 方差估计。

每层独立评价 N_s 行，包括零权重行：

\[
\widehat Z_s=N_s^{-1}\sum_i w_{si},\quad
\widehat Z=\sum_s\rho_s\widehat Z_s,\quad
r_{si}=\rho_sw_{si}/N_s,\quad W_{si}=r_{si}/\sum_{r,j}r_{rj}.
\]

父级 CSV 的后验权重为 W。各层 N 不同也成立；**漏掉 1/N_s 会把计算预算错误地变成先验质量**。建议保留 z、u/physical parameters、logL、logpi、logq、logw、stratum_id、全体 N_s 和 proposal 身份。对数合并为 `logw + log(rho_s) - log(N_s)`；分层数值缩放必须保留各层绝对 log-scale，不能先各自归一化再任意拼接。零权重可以不用于绘图，但证据与方差的分母仍是全部 N_s。

给定冻结 proposal、固定 N_s 和新 iid 评价随机数，Zhat_s 与 Zhat 分别是条件无偏的积分估计。`log(Zhat)`、分支概率和后验均值的比值估计**不是有限样本无偏**；有适当矩条件时一致。这里的 Z 是当前两个平坦均值边缘化约定下的固定目标积分，不因此获得新的跨模型 Bayes-factor 解释。

## 2. 证据、分支和均值的误差

令 s_s^2(v) 是层内全部评价行的无偏样本方差（ddof=1，N_s>=2）。独立层之间：

\[
\widehat{\operatorname{Var}}(\widehat Z)=\sum_s\rho_s^2s_s^2(w)/N_s,
\qquad \operatorname{SE}(\log\widehat Z)\simeq\operatorname{SE}(\widehat Z)/\widehat Z.
\]

对 h(theta)，Ahat= sum_s rho_s mean(w h)，mhat=Ahat/Zhat。一阶 ratio 方差为

\[
\widehat{\operatorname{Var}}(\widehat m)\simeq
\widehat Z^{-2}\sum_s\rho_s^2s_s^2\{w(h-\widehat m)\}/N_s.
\]

它同时适用于分支指示函数、任意物理参数均值和固定延迟点的总 TF 均值；协方差由上述 residual 的样本协方差替换样本方差得到。所有分支与均值必须复用同一分母的协方差，不能把分子和分母误差当独立。若 RSE(Z) 很大、少数行主导或有效尾部极少，delta-method 区间不可靠，应报告失效而不是只给一个小数位充分的误差。

层后验质量 Phat_r=rho_r Zhat_r/Zhat 的 Jacobian 为
`d Phat_r/d Z_s = rho_s * (1[r=s]-Phat_r)/Zhat`，与各层 Var(Zhat_s) 合成 covariance；其各行和应接近零。所有层都必须保留。

归一化权重的 Kish ESS=1/sum W^2 仅是权重集中度，不是“独立后验样本数”，也不是收敛证明。评价点在层内 iid，但不同层不具有相同分布；Kish 还受 N_s 分配影响。例如 L=1、q=pi 时证据误差恰为零，即使不成比例分配 N_s 使 Kish 小于总行数。还应报告每层 RSE(Z)、最大权重、权重最大的 1% 行贡献，以及每项关键 h 的 MCSE；不能从一个全局 ESS 推断所有尾部指标精度。

## 3. 防御分量与有限精度边界

由 q>=epsilon*pi，w<=L/epsilon，而且

\[
E_q[w^2]=\int L^2\pi^2/q\le\epsilon^{-1}E_\pi[L^2].
\]

因此“防御混合保证有限方差”需要 L 属于 L2(pi)，不是任意似然都自动成立。当前固定先验紧致、尺度下界正、幅度上界有限；两带均有数据且设计矩阵满列秩。实读 shared 输入各 42 点，最小测量标准差分别为 0.07352533475463842、0.07111622361471209。C>=diag(e^2)>0，且 C 有统一有限上界；`H.T C^-1 H` 也有统一正下界。故包含两项 log determinant 的均值边缘化 L 连续且有界。当前有界 h 和总 TF 亦满足所需矩条件。这个论证没有计算可用的 Lmax，**有限方差仍可能大得不可实用**，小 epsilon 也可能需要极多样本才能发现遗漏区域。

数值实现应在原始 z 上用 softplus/logsumexp 计算 logpi、全部混合 logq 和 logw。不要用 `log(u*(1-u))`，不要裁剪 z 后继续使用原 q，也不要把 expit 饱和行删除、重抽或直接赋零权重。当前 `core.validate_parameters` 允许 mu0==mu1、q==0，原条件变换可取闭 unit cube 的连续边界。因此显式启用 `allow_closed_unit=True` 后，可把有限 z 的自然 expit 舍入边界送入连续目标求值，密度比仍按原 z 计算。

这是一种**有限精度目标求值近似**，不是数学上精确计算了 T(expit(z))；必须记录饱和坐标/行数、其总归一化权重和最大单行权重，并比较 float64 与 longdouble logistic/transform 后转 float64 的 logL/logw 与总积分差。相应差异必须小于拟报告 MC 误差；不能仅以饱和次数小判定。Student-t 尾样本本身不应 fail-fast。真正非有限 z、错误的密度、目标线性代数数值失败应单独报错/诊断，不能冒充数学上的 L=0；合法零似然则保留为零权重。防御 logistic 抽样应使用明确的开区间 uniform 生成约定，避免 logit(0/1)，该零测集处理与重抽有限 t 尾样本不同。

源码依据：`/home/leonhard/software/MICA2-slowgp/archive/stability-v2-rslice/src/mica_slowgp/core.py:115-126,269-289`；条件变换见 `/home/leonhard/sci001/test/mica_slowgp/staged_stratified/src/mica_slowgp/nested.py:35-65`。上述有界性只针对当前具有正噪声的输入；core 通用接口允许零误差，因此不能把结论推广到任意输入。

## 4. 训练、评价和极小中心尺度

训练可以使用已经完成的 v3 子后验，但不得把训练行当作新 iid 评价样本直接加入估计。可将同层不同 seed 的归一化条件后验等权混合训练，避免按原行数或“最好看的 seed”选择；训练质量影响效率，不应改变积分目标。训练数据、拟合/正则化规则、mixture 参数、epsilon、nu、随机种子和全体评价 N_s 在评价前冻结。

不同评价 seed 可以共享一份冻结 proposal：给定 proposal，其独立新随机数产生独立积分估计，能够测量该 proposal 的 MC 误差。但这不能算多份独立 proposal 的覆盖检查；共同低效或很罕见的高权重区域仍可能全部未被遇到。预算加倍可以使用同一 seed 的样本前缀延长，但两预算因此相关。若根据已见评价结果重新拟合 proposal，必须登记新一轮并另用新评价样本；不能选择最有利的批次、丢弃失败 seed 或在 gate 首次通过时悄悄结束固定预算实验。

当前 `conditional_layer_diagnostic.md` 显示：低延迟层的 TF、分支和主体分布都未复现，不能只优化一个尾部指标；中间层负延迟面积占比的中心 q84-q16 约 1e-4，而均值受少量较大值影响。这适合在冻结训练阶段保留低层不同形状与中间层尾部的 proposal 分量，配合宽 scale 和防御项，并根据训练/独立 pilot 的方差配置各层预算。均值 h 的成本约束最优分配形式为 N_s proportional rho_s SD_q[w(h-m)]/sqrt(cost_s)，多指标时需采用预声明的保守综合准则，不能只为某项 gate 调参。

上述手段能降低同一指标的 MC 误差，无需扩大 IQR、改成 log 指标、截尾、平滑 CDF、重采样成等权行或改变 gate。精确加权 ECDF、原始 weighted quantiles 和原阈值继续使用。辅助误差可在每层对 **iid proposal 行等概率 bootstrap**，重新计算原权重、父级分母和 gate 指标；不能按后验权重抽样后假称原积分的 MC bootstrap。稀疏尾部/分位间隙下 bootstrap 也可能漏掉未见区域，应同时报告实际尾部行数和权重贡献。共享随机数可用于数值实现对照，不能用于三个独立评价 seed 的 gate 来人为降低差异。

## 5. 实现必须通过的 toy 与科学 pilot 记录

以下是验收要求，不表示已经完成这些测试。

1. **密度与 Jacobian**：低维可积目标验证 normalized t/logistic mixture、sample/logpdf 一致、logit Jacobian 符号正确；常数 L=1 应有 Z=1。多维 proposal 必须用共享 chi-square 缩放的标准 multivariate t，不可错误地逐坐标独立缩放后仍用 multivariate-t 密度。
2. **不等 N 的三层合并**：rho=(64,465,432)/961，层内 L=(1,2,4)，q=pi，N=(3,11,37)。必须得到 Z=2722/961、层后验质量=(64,930,1728)/2722，证据方差零；拼接样本后的 h 均值与解析结果一致。这个 toy 必须能识别遗漏 1/N 的实现。
3. **已知积分/后验的非恒定和多峰 toy**：用多次独立评价重复检查 Z、分支、均值、报告 MCSE 和经验离散度；包含训练 proposal 漏掉一峰但防御项仍覆盖的情形。不能要求单个有限随机估计等于真值，也不能将 ratio 宣称无偏。
4. **边界和生命周期**：测试 z=0、正负 40/1000、自然边界饱和与 longdouble 对照；检查全部行和计数保留、无 NaN、log-density 不受 expit 舍入污染。相同冻结 proposal/seed 可重现，save/load 不变；resume 后等于不中断同一随机流，错误 proposal/target/N 规则必须拒绝。
5. **科学 pilot 必报**：每层 N、实际目标调用、时间、Zhat/SE/RSE、层后验质量及误差、Kish/最大行和最大 1% 权重贡献、防御分量抽中数及贡献、饱和贡献、数值失败数、各关键分支/均值/总 TF 的 MCSE。proposal 训练调用和评价调用分别统计，不能把训练成本藏掉。
6. **科学有效性判断**：沿用三个独立评价 seed、两档预声明预算及现有原始 weighted gate；报告失败而不是调宽门槛。比较 seed 离散度是否与 MCSE 相容、预算增大是否降低估计噪声、rare-tail 是否仍被少数行主导。冻结 gate 的 Kish>=200 和最大行权重<=0.02 仍只是必要的权重分辨率检查。重要性采样、有限方差、较大 ESS 或小 SE 任一项单独都不能证明后验稳定、物理盘身份或距离/H0 无偏。

## 6. 本次实际核查范围

已用独立的短数组代数检查复核第 2 项的不等 N 合并，Z=2.832466181061394，层质量约 (0.02351212344,0.34166054372,0.63482733284)，与上述解析分数相符。它只检查合并代数，不算已通过完整 proposal/科学 pilot 测试。

只读审阅 `/home/leonhard/sci001/test/mica_slowgp/run_importance_pilot.py` 当前版本：`log(rho)+logw-logN`、全局 logsumexp、公共缩放后的证据方差和 `w*(h-mean)` ratio-MCSE 实现与本说明一致，未发现阻断级数学错误；物理数组、层号和权重的拼接对应正确。审阅时分支只有概率、层后验质量没有误差，未记录防御类别贡献、最大饱和行权重及 top-1% 权重贡献；这些诊断缺口已交 root 补充，本文件不修改该脚本。实际源码后续改动仍需由其实现方核验。
