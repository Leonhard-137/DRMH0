#!/usr/bin/env python3
"""Read-only posterior diagnostics for real Mrk817 two-component fits.

Writes new tables/figures only; never invokes MICA or changes fit products.
Run with the project's mica2 Python environment from any working directory.
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import yaml
from scipy.special import ndtr, gammainc
from scipy.stats import gamma, norm, spearmanr
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from plot_mica_transfer import read_component_parameters

BASE = ROOT / "test/mica_smooth"
OUT = BASE / "result/observed_second_component_statistics"
BANDS = ["UVM2", "UVW1", "U", "B", "V"]
CASES = ["raw", "responses_s20", "responses_s40", "all_s20", "all_s40"]
MODELS = ["gamma", "gaussian"]
QUANTILES = {"q025": .025, "q16": .15865, "q50": .5, "q84": .84135, "q975": .975}


def summarize(values):
    return dict(zip(QUANTILES, np.quantile(values, list(QUANTILES.values())).tolist()))


def analyze():
    rows, draws, integrals = [], {}, []
    for case in CASES:
        source = ROOT / "Mrk817_a" if case == "raw" else BASE / "cases_highpass" / case
        cfg = yaml.safe_load((source / "config/source_config.yaml").read_text())
        for model in MODELS:
            fit = dict(cfg[f"mica_{model}2_free"])
            fit["component_order"] = "moment1"
            for band in BANDS:
                run = source / f"runs/mica/{model}/free/2comp/run_UVW2_to_{band}_2comp_{model}"
                a, c, w, tau, _, nswap = read_component_parameters(run, fit)
                assert np.all(tau[0] <= tau[1])
                if model == "gamma":
                    std = np.sqrt(2) * w
                    neg = gammainc(2, np.maximum(-c / w, 0))
                    qlo, qhi = c + gamma.ppf(.05, 2) * w, c + gamma.ppf(.95, 2) * w
                else:
                    std = w
                    neg = ndtr(-tau / w)
                    qlo, qhi = c + norm.ppf(.05) * w, c + norm.ppf(.95) * w
                frac = a / a.sum(axis=0)
                broad_index = np.argmax(std, axis=0)
                col = np.arange(len(a[0]))
                metrics = {
                    "centroid0": tau[0], "centroid1": tau[1],
                    "std0": std[0], "std1": std[1], "width_native1": w[1],
                    "W90_1": qhi[1] - qlo[1], "lag_p05_1": qlo[1], "lag_p95_1": qhi[1],
                    "area1": a[1], "fraction1": frac[1],
                    "negative_fraction1": neg[1], "negative_total_from1": frac[1] * neg[1],
                    "negative_total_from0": frac[0] * neg[0], "negative_total": (frac * neg).sum(axis=0),
                    "std_ratio_1_to_0": std[1] / std[0],
                    "peak_ratio_1_to_0": (a[1] / w[1]) / (a[0] / w[0]),
                    "separation": (tau[1] - tau[0]) / np.sqrt(std[0]**2 + std[1]**2),
                    "std_broader": std[broad_index, col],
                    "fraction_broader": frac[broad_index, col],
                    "negative_total_from_broader": frac[broad_index, col] * neg[broad_index, col],
                }
                statefile = run / "run_mica_state.json"
                state = json.loads(statefile.read_text()) if statefile.exists() else {}
                if state and state["status"] != "complete":
                    raise RuntimeError(f"Incomplete run: {run}")
                rawsample = np.loadtxt(run / "data/posterior_sample1d.txt_2", ndmin=2)
                row = dict(case=case, model=model, band=band, run=str(run),
                           nposterior=len(a[0]), n_unique_draws=len(np.unique(rawsample, axis=0)),
                           nswapped=nswap, state=state.get("status", "legacy_no_state"),
                           target_saves=state.get("target_saves", 10000), ptol=fit["ptol"])
                for metric, values in metrics.items():
                    row.update({f"{metric}_{q}": v for q, v in summarize(values).items()})
                for f in [.02, .05, .1, .2]:
                    row[f"p_fraction1_gt_{f:g}"] = float(np.mean(frac[1] > f))
                for d in [10, 20, 40]:
                    row[f"p_std1_gt_{d}"] = float(np.mean(std[1] > d))
                    for f in [.02, .05, .1]:
                        mask = (std[1] > d) & (frac[1] > f)
                        row[f"p_std{d}_fraction{f:g}"] = float(np.mean(mask))
                        row[f"p_std{d}_fraction{f:g}_negative0.1"] = float(np.mean(mask & (neg[1] > .1)))
                row["p_std1_lt_2"] = float(np.mean(std[1] < 2))
                row["p_any_std20_fraction0.05"] = float(np.mean(np.any((std > 20) & (frac > .05), axis=0)))
                row["p_negative_fraction1_gt_0.1"] = float(np.mean(neg[1] > .1))
                row["p_negative_total_from1_gt_0.01"] = float(np.mean(frac[1] * neg[1] > .01))
                row["rho_std_fraction"] = float(spearmanr(std[1], frac[1]).statistic)
                # Width prior applies to theta for Gamma, sigma for Gaussian.
                width_lines = [x.split() for x in (run / "data/para_names_line.txt_2").read_text().splitlines() if "component_sigma" in x]
                upper = np.exp(float(width_lines[0][4]))
                row["native_width_prior_upper"] = float(upper)
                row["p_width1_gt_90pct_upper"] = float(np.mean(w[1] > .9 * upper))
                row["p_any_width_gt_90pct_upper"] = float(np.mean(w.max(axis=0) > .9 * upper))
                row["p_broader_component_is_C1"] = float(np.mean(broad_index == 1))
                # Independent numerical check of CDF convention at a real draw.
                i = len(a[0]) // 2
                if model == "gamma":
                    x = max(-c[1, i] / w[1, i], 0)
                    numeric = quad(lambda u: u * np.exp(-u), 0, x)[0]
                else:
                    x = -tau[1, i] / w[1, i]
                    numeric = quad(lambda u: np.exp(-u*u/2)/np.sqrt(2*np.pi), -np.inf, x)[0]
                integrals.append(abs(numeric - neg[1, i]))
                rows.append(row)
                draws[(case, model, band)] = metrics
    assert len(rows) == 50
    return rows, draws, dict(runs=50, states_complete=sum(r["state"] == "complete" for r in rows),
                            legacy_no_state=sum(r["state"] == "legacy_no_state" for r in rows),
                            max_CDF_quadrature_error=max(integrals),
                            sorted_by_centroid=True, posterior_weights="equal", posterior_cuts="none")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(rows):
    lookup = {(r["case"], r["model"], r["band"]): r for r in rows}
    style = Path("/mnt/c/Users/26330/.codex/skills/image-creator/assets/image-creator.mplstyle")
    # Local style copy makes this scientific artifact reproducible without personal paths.
    if not (OUT / "plot_style.mplstyle").exists():
        (OUT / "plot_style.mplstyle").write_text(style.read_text())
    categories = {
        "raw": ("Raw", "#222222", "o"),
        "responses_s20": ("Responses only, 20 d", "#E69F00", "^"),
        "responses_s40": ("Responses only, 40 d", "#009E73", "v"),
        "all_s20": ("W2 + responses, 20 d", "#1F4E9E", "s"),
        "all_s40": ("W2 + responses, 40 d", "#C43C39", "D"),
    }
    metrics = [("std1", r"Width $s_1$ (d)", 1),
               ("fraction1", r"Area fraction $f_1$ (%)", 100),
               ("negative_total_from1", r"Negative-lag contribution $b_1$ (%)", 100)]
    with plt.style.context(OUT / "plot_style.mplstyle"):
        for fname, cases in [("raw_vs_both_detrended", ["raw", "all_s20", "all_s40"]),
                             ("driver_processing_comparison", CASES[1:])]:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True, layout="constrained")
            for ri, model in enumerate(MODELS):
                for ci, (metric, label, scale) in enumerate(metrics):
                    ax = axes[ri, ci]
                    for j, case in enumerate(cases):
                        labelcase, color, marker = categories[case]
                        rs = [lookup[case, model, b] for b in BANDS]
                        median = np.array([r[f"{metric}_q50"] for r in rs]) * scale
                        lo = np.array([r[f"{metric}_q16"] for r in rs]) * scale
                        hi = np.array([r[f"{metric}_q84"] for r in rs]) * scale
                        x = np.arange(5) + (j-(len(cases)-1)/2)*.18
                        ax.errorbar(x, median, yerr=[median-lo, hi-median], fmt="none", ecolor="#B8B8B8", elinewidth=.8, capsize=0)
                        ax.plot(x, median, ls="none", marker=marker, ms=4.2, color=color, label=labelcase)
                    ax.set_ylabel(label)
                    ax.set_xticks(range(5), BANDS)
                    ax.xaxis.set_minor_locator(NullLocator())
                    ax.set_xlim(-.5, 4.5)
                    if metric == "std1":
                        ax.set_yscale("log")
                        ax.set_ylim(.1, 240)
                    else:
                        ax.set_ylim(0, 100 if metric == "fraction1" else 60)
                        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    ax.set_title(f"{model.title()}", fontsize=11)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside upper center", ncol=len(cases), fontsize=10)
            fig.savefig(OUT / f"{fname}.png", dpi=300)
            fig.savefig(OUT / f"{fname}.pdf")
            plt.close(fig)


def report(rows, audit):
    def interval(r, key, scale=1):
        v = [r[f"{key}_{q}"] * scale for q in ["q50", "q16", "q84"]]
        return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    lines = ["# Mrk817 真实数据第二组分后验统计", "",
        "来源：Mrk817_a 未去趋势基线及 cases_highpass 四组，5 波段 × 2 核 × 5 处理 = 50 组。排除旧低通 cases、模拟和单组分。",
        "", "## 口径", "",
        "每个等权联合后验样本按质心由小到大排序，C1 为较大质心组分；不删样本，不给等权后验再次加权。",
        "宽度 s1 使用核的标准差：Gaussian = 原生 sigma；Gamma-k2 = sqrt(2) theta。W90 是单个核包含 90% 面积的 5%–95% 分位跨度，不是参数的可信区间。",
        "f1=A1/(A0+A1)；n1=积分负时延的 C1 面积/A1；b1=f1*n1=负时延 C1 面积/(A0+A1)。所有比例均逐联合样本计算再取分位数，不能用边际中位数相乘代替。",
        "所有振幅有正值对数先验 [1e-5,10]。P(f1>5%) 等只表示在双组分模型内有一定面积的后验支持，不是双组分对单组分的检测显著性，也不是观测光变方差贡献。n1 或 b1 的中位数为零/舍入为零不代表整个后验都为零，必须结合区间。",
        "阈值 s1>10/20/40 d、f1>2/5/10%、n1>10% 为描述性分档，绝非物理判据。CSV 同时包含这些阈值的联合后验概率，保留连续量和 95% 区间供检查。",
        "所有时延为观测者系相对 W2 的时延。未截去 tau<0 的核；Gamma 的 onset=tau-2 theta，可在质心非负时仍有负时延面积。",
        "", "## 可比性与限制", "",
        "原始拟合冻结输入与 test/mica_smooth/data/original 完全一致；四组去趋势拟合与原始组各波段的原生参数名字、先验文件完全一致。40 组高通和5组原始 Gamma状态 complete；5组原始 Gaussian为旧输出，无状态文件，但有有限后验和原生日志的建模/后处理完成记录。",
        "原始 Gamma 的 UVW1/U/B 保存上限10000，其余原始 Gamma和高通 Gamma30000；ptol均0.01。Gaussian保存上限10000、ptol0.1。原始 UVM2 Gaussian后验365行，精度较有限。后验行数不是普遍可当作有效独立样本数。",
        "同一批观测被重复拟合，50项不是50个独立天体；跨处理的变化只作描述性比较，不当作独立样本的因果检验。去趋势保留原始误差，MICA仍用对角噪声协方差；下列区间是当前模型条件下的后验区间。",
        "", "## 归纳", "",
        "1. 原始未去趋势组普遍存在宽、面积不小、负时延占比不小的C1：以中位数 s1>20天、f1>5%、n1>10% 的描述性标准，10组中9组满足。双方去趋势后20天/40天分别均降至3/10。它们是同一数据的模型结果计数，不是独立样本显著性。",
        "2. 只处理响应的20组结果，每组后验都有至少一个 s>20天且面积占比>5%的核，概率为100%。宽度贴先验上界：原生宽度上界140.03天，Gamma标准差上界198.04天。Gamma中该宽核有时按质心排在C0，不能看到C1窄就称恢复正常。",
        "3. 双方去趋势后，UVM2的C1面积中位数约0.1%，P(f1>5%)仅5.0%–10.7%。其宽度68%区间仍可达0.4–67天，不能称为测得了一个窄第二组分。UVW1在20天也很弱，40天为约5.4%，该阈值概率仅57.5%–59.0%。",
        "4. U和B最稳定保留非忽略的第二组分面积。四种双方去趋势/核组合中，U的P(f1>5%)为92.6%–100%，B为98.1%–100%。这不自动等于物理BLR的检测。",
        "5. U的宽度和负时延强烈依赖核：Gamma的s1中位数7.9/16.2天、W90约24/50天，Gaussian为22.1/20.4天、W90约73/67天。Gamma20存在窄分支，40变得集中；因此U稳定的是有一定面积的C1，而非在所有核下都有同样超宽形状。",
        "6. B值得优先关注：Gaussian的b1原始16.1%、20天15.5%、40天16.1%，几乎没有减少。双方去趋势后宽且有面积且负时延占比>10%的联合后验概率分别91.8%、98.9%。Gamma B则有明显宽窄退化，宽度68%区间为1.5–87.4天和0.5–26.3天。U不是唯一持续异常波段。",
        "7. V依赖处理尺度和核：40天Gamma面积11.4%、Gaussian25.5%；Gaussian40的b1为8.0%，但68%区间0–14.5%，后验仍含窄分支。不能简单说sigma40让所有波段更差：Gamma B更窄，Gaussian B/V更宽，U更集中。",
        "8. 负时延面积与宽度/质心比存在数学联系，并非三条独立异常证据。Gaussian n1=Phi(-tau1/s1)；Gamma n1=G_k=2(max(2-sqrt(2)*tau1/s1,0))。相同正质心与标准差下两种核的负尾也可不同。",
        "9. 当前支持的归纳是：双方去趋势消除了大部分波段普遍需要的大面积宽核，但剩余结构有波段选择性和核函数依赖。它支持检查W2慢变的重要性，单靠本统计尚不能唯一识别独立慢变或盘+BLR；全局拟合改善也不能代替逐波段恢复验证。",
        "", "## 逐组结果", "", "中位数 [15.865%, 84.135%]。n1与b1均为百分比。", "",
        "|处理|模型|波段|s1 (d)|W90 (d)|f1 (%)|n1 (%)|b1 (%)|P(f1>5%)|",
        "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("|" + "|".join([r["case"], r["model"], r["band"], interval(r,"std1"),
            interval(r,"W90_1"), interval(r,"fraction1",100), interval(r,"negative_fraction1",100),
            interval(r,"negative_total_from1",100), f'{100*r["p_fraction1_gt_0.05"]:.1f}%']) + "|")
    lines += ["", "## 图", "", "图为中位数及 68.27% 等尾后验区间；宽度轴为对数，其余为百分比。颜色和点型仅区分数据处理，不赋予物理组分身份。", "",
              "![Raw versus both detrended](raw_vs_both_detrended.png)", "",
              "![Driver processing](driver_processing_comparison.png)", "", "## 数值核对", "", "```json", json.dumps(audit, indent=2), "```", ""]
    (OUT / "statistics.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _, audit = analyze()
    write_csv(OUT / "posterior_metrics.csv", rows)
    sensitivity = []
    for case in CASES:
        rs = [r for r in rows if r["case"] == case]
        for width in [10, 20, 40]:
            for frac in [.02, .05, .1]:
                key = f"p_std{width}_fraction{frac:g}_negative0.1"
                sensitivity.append(dict(case=case, width_threshold_days=width,
                    fraction_threshold=frac, negative_fraction_threshold=.1,
                    median_based_count=sum(r["std1_q50"] > width and r["fraction1_q50"] > frac and r["negative_fraction1_q50"] > .1 for r in rs),
                    conditional_probability_sum=sum(r[key] for r in rs),
                    n_runs_probability_gt_0_95=sum(r[key] > .95 for r in rs), n_runs=len(rs)))
    write_csv(OUT / "threshold_sensitivity.csv", sensitivity)
    (OUT / "validation.json").write_text(json.dumps(audit, indent=2))
    figures(rows)
    report(rows, audit)
    print(json.dumps(audit, indent=2))
    for r in rows:
        print(f'{r["case"]:15} {r["model"]:8} {r["band"]:4} s={r["std1_q50"]:6.2f} '
              f'f={100*r["fraction1_q50"]:5.1f}% n={100*r["negative_fraction1_q50"]:5.1f}% '
              f'b={100*r["negative_total_from1_q50"]:5.1f}% P(f>.05)={r["p_fraction1_gt_0.05"]:.3f} '
              f'P(wide,strong,neg)={r["p_std20_fraction0.05_negative0.1"]:.3f}')


if __name__ == "__main__":
    main()
