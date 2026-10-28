#!/usr/bin/env python3
"""Run the unified two-Gamma/two-Gaussian comparison for Mrk 509."""

from __future__ import annotations

import os
import runpy


os.environ["UNIFIED_SOURCE"] = "Mrk509"
runpy.run_path(
    "/home/leonhard/sci001/scripts/plot_ngc5548_gamma_selection.py",
    run_name="__main__",
)
