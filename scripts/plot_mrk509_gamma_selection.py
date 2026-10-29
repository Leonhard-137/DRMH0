#!/usr/bin/env python3
"""Run the unified two-Gamma/two-Gaussian comparison for Mrk 509."""

# 薄包装入口：设 UNIFIED_SOURCE=Mrk509 后通过 runpy 执行 NGC5548 命名的统一比较脚本。
# 实际源、静止系换算和输出路径由被调用脚本的 Mrk509 分支决定。
from __future__ import annotations

import os
import runpy


os.environ["UNIFIED_SOURCE"] = "Mrk509"
runpy.run_path(
    "/home/leonhard/sci001/scripts/plot_ngc5548_gamma_selection.py",
    run_name="__main__",
)
