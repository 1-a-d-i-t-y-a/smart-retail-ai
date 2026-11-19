#!/usr/bin/env python3
"""
scripts/build_all_models.py
---------------------------
Unified end-to-end ML build pipeline for Smart Retail AI:
1. Generates deterministic training and evaluation datasets (seed 42).
2. Trains all baseline ML models (Product HOG+SVM, Sentiment TF-IDF+LR, Face DB).
3. Executes rigorous model evaluation and saves verified metrics to artifacts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_datasets import main as generate_datasets_main
from scripts.train_baseline_models import main as train_models_main
from scripts.evaluate_models import main as evaluate_models_main


def main():
    print("=" * 70)
    print("SMART RETAIL AI — UNIFIED ML PIPELINE (BUILD & EVALUATE)")
    print("=" * 70)
    generate_datasets_main()
    print()
    train_models_main()
    print()
    evaluate_models_main()
    print("=" * 70)
    print("ALL ML ARTIFACTS AND EVALUATION SUITES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
