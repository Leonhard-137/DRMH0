# 包级版本标记。导入这里只定义 MODEL_VERSION；具体拟合由 __main__.py 显式调度。
# 该版本名描述保留的 shared-SED 基线；新增 H0 四情景由 comparison.py 单独进入。
"""Shared standard-SED FFA, one luminosity distance per AGN, fixed-f0 H0.

This preserves the 2026-09-04 trial, including its documented approximations.
"""
MODEL_VERSION = "shared-standard-sed-no-f0-v1"
