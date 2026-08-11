#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from ffa import nuc


def main():
    ap = argparse.ArgumentParser(
        description="Run nuclear reddening and intrinsic SED analysis."
    )
    ap.add_argument("source")
    args = ap.parse_args()

    nuc.run(Path(args.source))


if __name__ == "__main__":
    main()