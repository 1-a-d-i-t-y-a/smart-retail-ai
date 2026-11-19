# Final Project Evidence Package — Smart Retail AI v1.2.0

**Generated:** 2026-11-19  
**Branch:** `engineering/reproducibility-and-release`

---

## Automated Test Results (actual execution)

| Metric | Result |
|---|---|
| Tests collected | **129** |
| Tests passed | **129** |
| Tests failed | **0** |
| Test duration | ~5.8 s |
| Python version | 3.12 (host execution environment) |

---

## Model Evaluation (actual run results)

| Component | Method | Train samples | Test samples | Test accuracy | Challenge samples | Challenge accuracy |
|---|---|---:|---:|---:|---:|---:|
| Product classification | HOG + RBF-kernel SVM | 240 | 60 | 100.0% | 20 | 100.0% |
| Sentiment analysis | TF-IDF + Logistic Regression | 283 | 95 | 100.0% | 20 | 55.0% |
| Face verification | Haar Cascade + SSIM/Pearson | — | 3 cases | 100.0% | 1 functional distractor | Passed |

> **Evaluation Notice:** The benchmark datasets are synthetic or controlled fixtures deliberately generated for reproducibility and privacy. Achieved 100% accuracy on the controlled synthetic held-out benchmark demonstrates reproducible behavior on generated data and should not be interpreted as real-world retail generalization. Challenge results are intentionally harder than the controlled test split and expose generalization limits on independently generated/hand-written samples.

---

## Face Verification Scenario Breakdown

| Scenario | Type | Expected | Actual | Correct |
|---|---|---|---|---|
| Reference image (self-match) | Accuracy case | matched | matched | ✅ |
| Positive query (calibrated brightness shift) | Accuracy case | matched | matched | ✅ |
| Negative (different face) | Accuracy case | unmatched | unmatched | ✅ |
| No-face distractor | **Functional test** | no-face message | no-face message | ✅ |

> The no-face distractor is a **functional test** (verifies the "no face detected" code path), not counted in the 3/3 accuracy denominator.

---

## Security and Packaging

| Check | Result |
|---|---|
| Secret scan (regex patterns) | PASS — no secrets found |
| Archive path traversal (`../`, `..\`) | PASS — zero traversal sequences |
| Archive absolute paths | PASS — none |
| Windows drive paths (`C:`, `D:`) | PASS — none |
| UNC network paths (`//`, `\\`) | PASS — none |
| Archive symlinks | PASS — none |
| Exclusion rules verified | See `artifacts/packaging/exclusion_test_results.txt` (ALL RULES PASSED) |
| SHA-256 | `ee2f24bbf03bd223b3eb97186c328c83e31f7d2dff606043a2cb338ccef75666` |

---

## Upload Validation

| Feature | Implementation |
|---|---|
| Bounded upload reading | `read_bounded_upload()` — 64 KiB chunks, halts at `max_bytes + 1` |
| HTTP 413 response | `/classify-product` and `/recognize-face` return 413 for oversized uploads |
| Decompression-bomb protection | Pillow `Image.verify()` and header dimensions check before OpenCV decode |
| Strict Content-Type Allowlist | `image/jpeg`, `image/png`, `image/webp`, `image/bmp` (normalized) |
| Dimension bounds | 10×10 px minimum, 4096×4096 px maximum |
| Byte limit | 5 MB default (configurable via `SMART_RETAIL_MAX_IMAGE_BYTES`) |

---

## Docker Status

Docker image build and runtime were **not executed** because Docker CLI was unavailable in the local Windows environment. The `Dockerfile` and `docker-compose.yml` have been inspected and verified against:
- Architecture: Single-stage build
- Base image: `python:3.12-slim`
- Non-root user: `appuser`
- Port: `8000`
- Health check: `GET /health/ready`
- Build step: `python scripts/build_all_models.py`
- Compose service: `smart-retail-api`

---

## Evidence Map

| Artifact directory | Contents |
|---|---|
| `artifacts/product/` | HOG+SVM metrics JSON, confusion matrix |
| `artifacts/face/` | Evaluation JSON, annotated reference images |
| `artifacts/sentiment/` | Metrics JSON, classification report, confusion matrix PNG |
| `artifacts/chatbot/` | Intent evaluation, smoke test JSON |
| `artifacts/integration/` | Unified pipeline smoke test |
| `artifacts/testing/` | Error handling evidence |
| `artifacts/evaluation/` | Unified `model_evaluation.json` (authoritative) |
| `artifacts/api/` | OpenAPI JSON, documentation screenshot |
| `artifacts/deployment/` | Deployment configuration evidence |
| `artifacts/end_to_end/` | End-to-end pipeline evaluation |
| `artifacts/packaging/` | Exclusion rule test results (ALL RULES PASSED) |
| `artifacts/final/` | This file, `final_evidence_summary.json` |

---

## Reporting Note

All artifacts are **reconstructed/re-implemented evidence** based on the project specification and synthetic datasets generated with seed 42. They must not be described as original historical internship screenshots, logs, datasets, or telemetry unless independently supported by the user's original records.
