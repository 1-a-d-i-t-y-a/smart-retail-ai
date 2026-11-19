#!/usr/bin/env python3
"""
scripts/build_all_models.py
---------------------------
Entrypoint for Docker and CI model artifact build.
Calls train_baseline_models.main().
"""

from __future__ import annotations

from scripts.train_baseline_models import main

if __name__ == "__main__":
    main()
