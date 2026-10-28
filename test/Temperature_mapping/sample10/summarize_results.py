"""Render evidence-backed Chinese handoff without assigning physical labels."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"


def fmt(x, digits=3):
    return "未运行" if x is None else f"{x:.{digits}f}"


def main():
    rows=json.loads((OUT/"summary.json").read_text())
    # Input arrays are frozen during the run; later provenance clarifications
    # update descriptions only. Preserve computation-derived warnings.
    manifest=json.loads((ROOT/"derived/input_manifest.json").read_text())
    entries={r["source"]:r for r in manifest["sources"]}
    physical=json.loads((ROOT/"physical_parameters.json").read_text())
    computational_prefixes=("Only six (or fewer)", "Monitoring spans fewer", "All-band common", "At least one band", "99th percentile", "Maps are regularized")
    for row in rows:
        warnings=entries[row["source"]].get("warnings",[])+physical["sources"][row["source"]].get("warnings",[])
        warnings += [w for w in row["quality_warnings"] if w.startswith(computational_prefixes)]
        if row["coverage"]["common_span_days"] < 112:
            warnings.append("All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.")
        row["quality_warnings"]=list(dict.fromkeys(warnings))
        (OUT/row["source"]/"summary.json").write_text(json.dumps(row,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    (OUT/"summary.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    status=json.loads((OUT/"batch_status.json").read_text())
    validation=json.loads((OUT/"numerical_validation.json").read_text())
    fits=sum(len(json.loads((OUT/r["source"]/"metrics.json").read_text())) for r in rows)
    by_source={r["source"]:r for r in rows}
    assert len(by_source)==10, "The ten-source comparison report requires all ten completed sources"
    slow_summary="、".join(f"{s} {by_source[s]['diagnostics']['low28_q_rms_ratio']:.3f}" for s in ("Mrk817_a","F9_a","Mrk279_b","Mrk509"))
    swift_correlations=[by_source[s]["diagnostics"]["swift6_map_correlation"] for s in ("Mrk817_a","Mrk142","Mrk335","NGC5548")]
    short_spans="、".join(f"{by_source[s]['time_span_days']:.0f}" for s in ("NGC4151_a","NGC6814","NGC2617_a"))
    short_common="、".join(f"{by_source[s]['coverage']['common_span_days']:.0f}" for s in ("Mrk142","Mrk335"))
    lines=["# 主要十源：温度映射与慢／快变化复现测试", "",
        f"实际完成 {len(rows)}/10 源，保存 {fits} 组拟合结果；失败 {len(status['failures'])} 源。", "",
        "样本严格来自 `DRMH0/tables/sample_observations.csv`；不同后缀是特定观测期，不将同一 AGN 当作多个独立源。完整 Mrk817 与此前 NGC5548 的独立复现结果保留在上级目录。", "",
        "## 这一轮首先看到了什么", "",
        f"- 在本轮固定模型和比较区域下，28 天低通温度场的 RMS／原始温度场 RMS 为：{slow_summary}；NGC5548 为 {by_source['NGC5548']['diagnostics']['low28_q_rms_ratio']:.3f}。这只是拟合场的慢变化保留程度，不是慢波存在的概率，也不能作为跨源物理强度排名。",
        f"- 在四个具有额外 HST／地面波段的源中，仅 Swift 六波段图与全波段图的相关为 {min(swift_correlations):.3f}–{max(swift_correlations):.3f}。全波段约束确实影响恢复图样；另外六个源本来只有 Swift，表中的 1.000 是相同输入的恒等比较。",
        f"- 去掉 U/u 后，NGC5548 的图相关为 {by_source['NGC5548']['diagnostics']['without_U_map_correlation']:.3f}，Mrk817_a 为 {by_source['Mrk817_a']['diagnostics']['without_U_map_correlation']:.3f}；它们对这些波段的敏感性不同，但这里尚未分离 BLR。",
        f"- NGC4151_a、NGC6814、NGC2617_a 的总时长只有约 {short_spans} 天；Mrk142 和 Mrk335 的全波段共同时间包络也只有约 {short_common} 天。56 天滤波结果都需要特别谨慎。", "",
        "这些是首轮条件性结果，尚未进行传播轨迹显著性或不同物理机制的注入检验。", "",
        "## 先看哪些文件", "",
        "- `outputs/temperature_mapping_atlas.pdf`：全部图的 PDF 图册。",
        "- `outputs/figures/`：每幅图的矢量 PDF、300 dpi PNG 与 JPEG 预览。",
        "- `outputs/summary.csv`：逐源主拟合数值。",
        "- `outputs/<source>/fits.npz`：原始数据、预测、径向核及每种测试的温度场。",
        "- `outputs/<source>/metrics.json`、`summary.json`、`band_residuals.json`：各测试指标、条件和逐波段残差。",
        "- `DATA_PROVENANCE.md`、`derived/input_manifest.json`：数据、单位、时标、误差修正与缺口。",
        "- `PARAMETERS.md`、`physical_parameters.json`：固定盘参数和文献来源。", "",
        "## 冻结的分析方法", "",
        r"在给定薄盘 $T_0(R)$、黑洞质量、距离、倾角和吸积率的条件下，用线性化黑体核将 $q=\delta T/T_0$ 积分为多波段光变；每个波段另有一个不受惩罚的常数偏置。最小化带观测误差权重的残差平方和与 NK22 型幅度、径向及时间差分惩罚之和。", "",
        "主网格为 50 个径向格、250 个时间格，主正则化 ξ=10；ξ=1、100、1000 为固定对照，不按图样好看程度或理论预期选择。径向范围依源的文献设置确定，因此跨源相同格数也不代表同一径向分辨率。", "",
        "| 测试 | 操作 | 能检查什么 |",
        "|---|---|---|",
        "| 原始光变 | ξ=1/10/100/1000 | 正则化敏感性 |",
        "| 低通 | 每个波段先插值到 0.1 天网格，再用 FWHM=7/28/56 天高斯平滑，保留原始权重反演 | 慢变化对应的正则化温度场 |",
        "| 高通 | 原始光变减去对应的低通结果，再反演 | 去掉该慢变化后剩余图样 |",
        "| 时间网格 | ξ=10，Nt=100/250/500 | 离散化及惩罚变化的联合敏感性，不等于严格收敛检验 |",
        "| 去掉 U/u | 排除中心波长 3200–3800 Å 的波段，保持相同 R、t 网格和 ξ | 这些波段对结果的影响，不证明其他波段无 BLR 污染 |",
        "| 仅 Swift 六波段 | 唯一识别六滤光片后重拟合；本来仅六波段时直接标记与主拟合相同 | 增加其他波段是否改变图样 |", "",
        "没有把 MICA 分解结果当作输入；没有联合 MICA 与温度场，也没有对十源运行残差时延 MCMC、传播速度拟合或检出显著性标定。", "",
        "## 逐源主结果", "",
        "下表 q99 是全图 |δT/T| 的第 99 百分位，不是误差条、显著性或盘变动的独立检测统计量。χ²/N 是每个数据点的加权残差平方，不是除以自由度的约化 χ²。", "",
        "| 源 | 波段 | 数据点 | 总跨度/d | Δt/d | χ²/N | q99 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['source']} | {r['n_bands']} | {r['n_data']} | {r['time_span_days']:.1f} | {r['dt_days']:.3f} | {r['chi2_per_point']:.3f} | {r['q99']:.3f} |")
    lines += ["", "## 稳定性与 28 天分解", "",
        "相关系数只描述选定比较区域的两张拟合图是否相似，没有传播方向或显著性含义。通常使用所有波段时间包络的交集以及平均归一化径向核累计 5%–95% 的径向格；该区域仍可包含观测缺口，也不等于有效分辨率或置信区间。若交集不足则回退全图并在单源 JSON 中明示。", "",
        "低／高通 RMS 比是对应温度场与原始拟合温度场的 RMS 之比；两者相关，不能解释成可相加的物理功率占比。", "",
        "| 源 | 去 U/u 图相关 | 仅 Swift 图相关 | Nt=100 图相关 | Nt=500 图相关 | 低通28 RMS比 | 高通28 RMS比 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        d=r["diagnostics"]
        values=[d.get(k) for k in ("without_U_map_correlation","swift6_map_correlation","nt100_map_correlation","nt500_map_correlation","low28_q_rms_ratio","high28_q_rms_ratio")]
        lines.append("| "+r["source"]+" | "+" | ".join(fmt(v) for v in values)+" |")
    lines += ["", "## 逐源限制与需要优先检查的点", ""]
    for r in rows:
        c=r["coverage"]
        maxgap=max(x["max_gap_days"] for x in c["bands"])
        lines += ["### "+r["source"], "",
            f"全波段时间包络交集为 {c['common_span_days']:.1f} 天；最大单波段缺口 {maxgap:.1f} 天。所有时间宽度均为观测者系。", ""]
        lines += ["- "+x for x in r["quality_warnings"]]
        lines += ["", "比较区域："+r["comparison_region"]+"。", ""]
    lines += ["## 验证与计算成本", "",
        f"独立小规模稠密解与生产稀疏解的温度最大差为 {validation['dense_q_max_abs_error']:.3g}；新增可选径向起点／固定时间边界在默认设置下产生的前向矩阵与旧版完全相同。", ""]
    errors=[]
    for r in rows:
        v=json.loads((OUT/r["source"]/"validation.json").read_text())
        assert v["passed"]
        errors.extend(float(x) for k,x in v.items() if k.endswith("max_abs_error"))
    lines += [f"十源检查包括正常方程残差、固定偏置不改变温度场，以及每个平滑宽度下 q_low+q_high=q_original 的数值线性恒等式；上述温度／预测检查中的最大绝对偏差为 {max(errors):.3g}。线性恒等式验证实现，不是物理独立证据。", "",
        f"顺序单线程拟合累计 {sum(r['fit_seconds'] for r in rows):.1f} 秒（不含资料核查、绘图和人工检查）；逐源计时保存在 summary.csv。", "",
        "## 解释边界", "",
        "1. **得到条纹不等于检测到传播。** 这里的观测量是多波段积分光度；温度场仍是欠定反问题的正则化解。未提供像素后验或不同传播模型的校准检验。",
        "2. **平滑改变噪声协方差。** 按原文对照保留原始误差作为权重，因此低／高通 χ² 仅作描述，不能用独立高斯似然显著性解读。边界采用常值延拓，缺口用线性插值。",
        "3. **时间跨度不同。** 短于 112 天的监测甚至不能覆盖两个 56 天平滑宽度；没有长条纹不能作为慢变化不存在的证据。",
        "4. **ξ 相同不代表相同物理先验强度。** 数据误差、波段核、半径尺度和 Δt 均不同；Nt 扫描也改变未按物理导数归一化的差分惩罚。",
        "5. **固定物理模型不等于真实盘。** 质量、距离、吸积率、倾角以及未统一去消光等选择影响径向核和振幅；超爱丁顿目标的薄盘解应视为形式测试。没有把这些不确定性传播到图中。本轮不额外去消光；尤其 NGC2617_a 的上游星等转流量及是否预先去红化尚未独立重建，绝对温度幅度只能在本地单位约定成立的条件下解释。",
        "6. **U/u 去除不构成 BLR 分离。** 共模或宽光谱的 BLR／其他非盘变化仍可被温度模型吸收。",
        "7. **这是方法迁移与数据条件下的复现测试。** 不是声称每个观测期都有一篇完全相同输入的发表温度图可供逐像素复现。", "",
        "## 方法来源", "",
        "- [Neustadt & Kochanek 2022](https://arxiv.org/abs/2201.10565v2)：线性温度反演与多源应用。",
        "- [AGN STORM 2 VI](https://arxiv.org/html/2310.01497)：7/28/56 天平滑与相减对照。",
        "- 本次数据和盘参数的逐项来源见同目录 DATA_PROVENANCE.md、PARAMETERS.md。", ""]
    (ROOT/"RESULTS.md").write_text("\n".join(lines))
    print("Saved", ROOT/"RESULTS.md")


if __name__=="__main__":
    main()
