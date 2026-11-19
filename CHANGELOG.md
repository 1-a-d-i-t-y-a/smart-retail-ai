# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-19

### Added
- Reconstructed `data/chatbot/retail_knowledge.json` for deterministic chatbot intent resolution.
- Added reproducible training script `scripts/train_baseline_models.py` generating HOG+SVM product classifier, TF-IDF sentiment classifier, and face database.
- Added `scripts/build_all_models.py` for automated Docker and CI/CD artifact building.
- Added 24 automated Pytest unit and integration tests under `tests/`.
- Added GitHub Actions continuous integration workflow (`.github/workflows/ci.yml`).
- Added comprehensive documentation: `docs/ARCHITECTURE.md`, `docs/API_REFERENCE.md`, `docs/MODEL_EVALUATION.md`, `docs/DEVELOPMENT.md`.
- Added standard community health files: `CODE_OF_CONDUCT.md`, `SECURITY.md`, and issue templates.

### Fixed
- Fixed fatal startup crash (`FileNotFoundError`) caused by missing chatbot knowledge base file.
- Fixed OpenCV `detectMultiScale` assertion error by bundling fallback Haar Cascade XML in `data/haarcascades/`.
- Removed tracked `.pyc` and `__pycache__` artifacts from git index and committed `.gitignore`.
