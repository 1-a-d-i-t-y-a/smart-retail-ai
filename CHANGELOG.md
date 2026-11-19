# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-11-19

### Added
- `app/utils/validation.py`: `read_bounded_upload()` async helper reads in 64 KiB chunks and raises `PayloadTooLargeError` immediately on exceeding the byte limit — prevents full-body buffering of oversized uploads.
- `app/utils/validation.py`: Pillow `Image.verify()` header inspection added to `validate_image_payload()` for decompression-bomb protection before OpenCV full decode.
- HTTP **413 Request Entity Too Large** responses in `/classify-product` and `/recognize-face` for uploads exceeding `settings.max_image_bytes`.
- `scripts/package_release.py`: Explicit exclusion set (`EXCLUDE_DIRS`, `EXCLUDE_FILENAMES`, `EXCLUDE_EXTENSIONS`) independent of walk filtering.
- `scripts/package_release.py`: Secret scanner using regex patterns for GitHub PATs, OpenAI keys, AWS access keys, PEM blocks, and password/token/secret assignments — aborts packaging without producing the archive on any match.
- `scripts/package_release.py`: Archive path-traversal and symlink validation step (Phase 2 of release pipeline).
- `artifacts/packaging/exclusion_test_results.txt`: Automatically generated exclusion rule verification report.
- `tests/test_packaging.py`: 20 unit tests covering exclusion rules, secret scanner, and packaging constant integrity.
- `tests/test_validation.py`: Tests for `read_bounded_upload()` (within-limit, over-limit, empty-file).
- `tests/test_product.py`: HTTP 413 test for oversized product uploads.
- `tests/test_face.py`: HTTP 413 test for oversized face uploads; replaced permissive `assert status_code in (200, 503)` with deterministic 503 test using the reference image.
- `CHANGELOG.md`: Formal changelog entry for v1.2.0.

### Changed
- README: Replaced "Production-Grade Reliability" with "Defensive Reliability" and "Pinned Python dependencies" with "Version-constrained Python dependencies".
- README, `docs/TESTING.md`: Updated test count badge and reference from 52 to 57 passing tests.
- `app/api/routes_product.py` and `routes_face.py`: Switched from `await file.read()` to `await read_bounded_upload()` for bounded streaming reads.

### Fixed
- `tests/test_face.py`: Removed permissive `assert response.status_code in (200, 503)` — replaced with a deterministic test that supplies the registered reference image, guaranteeing a DB lookup and therefore a 503 on missing database.
- `tests/test_product.py`: Corrected corrupt-image assertion to accept both `"Could not decode image"` and `"header validation failed"` after Pillow pre-check was added.

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
