# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-11-19

### Added
- `app/utils/validation.py`: Bounded upload reader `read_bounded_upload()` that streams uploads in 64 KiB chunks, strictly caps reads at `max_bytes + 1` bytes, and immediately raises `PayloadTooLargeError` without buffering excess data.
- `app/utils/validation.py`: Strict Content-Type allowlist normalization (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`) with rejection of unsupported MIME types (e.g. `image/svg+xml`).
- `app/utils/validation.py`: Format compatibility validation matching Content-Type against Pillow header format detection.
- `app/utils/validation.py`: Pre-decode dimension validation inspecting image dimensions from headers prior to full OpenCV matrix decoding.
- `app/utils/validation.py`: Decompression-bomb defense using `warnings.catch_warnings()` catching `Image.DecompressionBombWarning` and `Image.DecompressionBombError` as structured `ValueError`s.
- HTTP **413 Request Entity Too Large** responses in `/classify-product` and `/recognize-face` for oversized uploads.
- `data/fixtures/face/`: Tracked canonical facial input fixtures (`canonical_reference.png`, `canonical_negative.png`) with documented cryptographic provenance, completely decoupling dataset generation from output artifacts.
- `scripts/package_release.py`: Pre-generation of exclusion verification report (`artifacts/packaging/exclusion_test_results.txt`), expanded secret scanner detecting fine-grained PATs (`github_pat_`), classic PATs (`ghp_`), OAuth tokens (`gho_`), OpenAI keys, AWS access keys, PEM blocks, and JSON assignments with placeholder filtering.
- `scripts/package_release.py`: Comprehensive archive integrity validation rejecting absolute paths, Windows drive letters, UNC network paths, path traversal sequences (`../`, `..\`), and symlinks.
- `.env.example`: Canonical configuration template documenting environment variables and upload limits.
- `.github/workflows/ci.yml`: Automated CI workflow executing Python 3.11 and 3.12 test jobs, model building, release distribution packaging, and artifact publishing (`actions/upload-artifact@v4`).
- `tests/test_packaging.py`: 63 unit tests covering path exclusion rules, scannability, secret pattern detection, archive traversal/drive path validation, and set integrity.
- `tests/test_validation.py`: 20 unit tests covering normalized MIME types, format mismatch, decompression bomb, and bounded streaming reads.

### Changed
- README: Replaced "Production-Grade Reliability" with "Defensive Reliability" and clarified synthetic benchmark limits.
- README, `docs/TESTING.md`: Synchronized test suite metrics to 129 passing automated tests (100% pass rate).
- `docs/DEPLOYMENT.md`: Corrected single-stage Docker build documentation, updated non-root user to `appuser`, updated healthcheck endpoint to `/health/ready`, and corrected Compose service name to `smart-retail-api`.
- `FINAL_RELEASE_REPORT.md`: Comprehensive release report updated with exact SHA-256 checksum, package size, and test breakdown.

### Fixed
- `tests/test_face.py`: Replaced permissive assertion with deterministic 503 check using registered reference image.
- `tests/test_product.py`: Aligned corrupt image decoding assertions with Pillow header validation errors.
- `scripts/evaluate_models.py`: Added `sys.path` resolution for seamless CLI execution and separated non-face distractor check as a functional test.

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
