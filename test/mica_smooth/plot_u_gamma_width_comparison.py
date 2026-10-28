#!/usr/bin/env python3
"""Display existing, centroid-ordered U Gamma posteriors; no refitting."""
from pathlib import Path
import sys
import json
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from plot_mica_transfer import read_component_parameters, kernel

OUT = ROOT / "test/mica_smooth/result/observed_second_component_statistics"
LAG = np.linspace(-60, 160, 2201)
summary = {}
ymax = 0.0
with plt.style.context(OUT / "plot_style.mplstyle"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for sigma, color in [(20, "#1F4E9E"), (40, "#C43C39")]:
        case = ROOT / f"test/mica_smooth/cases_highpass/all_s{sigma}"
        fit = yaml.safe_load((case / "config/source_config.yaml").read_text())["mica_gamma2_free"]
        fit = dict(fit, component_order="moment1")
        run = case / "runs/mica/gamma/free/2comp/run_UVW2_to_U_2comp_gamma"
        a, c, w, tau, _, _ = read_component_parameters(run, fit)
        # Process lag blocks so the diagnostic does not retain a large draw grid.
        quantiles = np.hstack([np.quantile(kernel(l, a[1], c[1], w[1], "gamma"),
                           [.025, .15865, .5, .84135, .975], axis=0)
                           for l in np.array_split(LAG, 12)])
        ymax = max(ymax, float(quantiles[-1].max()))
        label = rf"$\sigma_{{\rm filter}}={sigma}$ d"
        for low, high, alpha in [(0, 4, .10), (1, 3, .25)]:
            axes[0].fill_between(LAG, quantiles[low], quantiles[high], color=color, alpha=alpha, lw=0)
        axes[0].plot(LAG, np.where(quantiles[2] > 0, quantiles[2], np.nan), color=color, lw=1.5, label=label)
        am, tm, wm = [np.median(x[1]) for x in [a, tau, w]]
        curve = kernel(LAG, am, tm - 2*wm, wm, "gamma")[0]
        axes[1].plot(LAG, np.where(curve > 0, curve, np.nan), color=color, lw=1.5, label=label)
        logstd = np.log10(np.sqrt(2)*w[1])
        bins = np.linspace(-.8, 2.3, 48)
        density, edges = np.histogram(logstd, bins=bins, density=True)
        axes[2].stairs(density, 10**edges, color=color, lw=1.4, label=label)
        summary[str(sigma)] = {
            "std_16_50_84_975_days": np.quantile(10**logstd, [.15865,.5,.84135,.975]).tolist(),
            "P_std_lt2": float(np.mean(10**logstd < 2)),
            "P_std_gt40": float(np.mean(10**logstd > 40)),
        }
    for ax in axes[:2]:
        ax.set_xlim(-60, 160)
        ax.set_ylim(1e-6, 1.15*ymax)
        ax.set_yscale("log")
        ax.set_xlabel(r"Lag $\tau$ (d)")
        ax.set_ylabel(r"$\Psi_1(\tau)$ (MICA units)")
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.axvline(0, color="#D9D9D9", lw=.8, zorder=0)
    axes[0].set_title("Pointwise posterior")
    axes[1].set_title("Marginal-median parameters")
    axes[2].set_title("Width posterior")
    axes[2].set_xlabel(r"Standard deviation $s_1$ (d)")
    axes[2].set_ylabel(r"Density per $\log_{10}s_1$")
    axes[2].set_xscale("log")
    axes[2].set_xlim(10**(-.8), 10**2.3)
    axes[2].set_ylim(bottom=0)
    axes[2].yaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    fig.savefig(OUT / "U_gamma_sigma20_vs40.png", dpi=300)
    fig.savefig(OUT / "U_gamma_sigma20_vs40.pdf")
    plt.close(fig)
(OUT / "U_gamma_sigma20_vs40.json").write_text(json.dumps(summary, indent=2))
(OUT / "U_gamma_sigma20_vs40_caption.md").write_text(
    "真实 Mrk817 U，W2 与响应同时去趋势，C1逐联合样本按质心排序。蓝色sigma20、红色sigma40。"
    "左：逐时延点的后验中位数、68.27%深色和95%浅色等尾可信带；可信带不是单个核的形状。"
    "中：将面积、质心、宽度各自边际中位数代入Gamma-k2，作为代表性曲线，不宣称是联合后验最高概率解。"
    "右：标准差的等权后验直方图，纵轴为每log10标准差的概率密度。"
    "使用所有后验样本，无分支筛选，无重新拟合。\n")
print(json.dumps(summary, indent=2))
