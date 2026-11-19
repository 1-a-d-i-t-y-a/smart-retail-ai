# Smart Retail AI Platform: Developer Guide

This guide details local environment setup, ML pipeline execution, testing, and contribution standards for developers working on the Smart Retail AI platform.

---

## 1. Prerequisites

- **Python:** 3.11 or 3.12 (tested on 3.12.10).
- **Git:** 2.30+.
- **Virtual Environment:** `venv` or `conda`.
- **System Dependencies:** Standard C runtime libraries (required by OpenCV and NumPy).

---

## 2. Quickstart Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/1-a-d-i-t-y-a/smart-retail-ai.git
cd smart-retail-ai
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Build Models & Generate Datasets
Run the unified build pipeline to generate synthetic datasets, train all baseline models, and compute evaluation metrics:
```bash
python scripts/build_all_models.py
```
This produces:
- `models/product_classifier.joblib` (HOG + RBF SVM)
- `models/sentiment_classifier.joblib` (TF-IDF + Logistic Regression)
- `models/face_db.pkl` (Face verification template)
- Evaluation reports in `artifacts/`

### Step 5: Launch the Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000/docs` to interact with Swagger UI.

---

## 3. Running Automated Tests

Run the full pytest suite (52 tests covering API routes, ML pipelines, error handling, and validation):
```bash
# Run all tests verbosely
pytest -v

# Run with test coverage
pytest --cov=app tests/

# Run a specific module
pytest tests/test_product.py -v
```

---

## 4. Environment Configuration

All settings support environment variable overrides with defensive defaults:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SMART_RETAIL_APP_NAME` | `AI-Powered Smart Retail & Customer Intelligence Platform` | API title |
| `SMART_RETAIL_APP_VERSION` | `1.2.0` | Application semantic version |
| `SMART_RETAIL_ENV` | `development` | Deployment environment |
| `SMART_RETAIL_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`) |
| `SMART_RETAIL_MAX_IMAGE_BYTES` | `5242880` (5 MB) | Maximum uploaded image payload |
| `SMART_RETAIL_MIN_IMAGE_DIM` | `10` | Minimum image width/height in pixels |
| `SMART_RETAIL_MAX_IMAGE_DIM` | `4096` | Maximum image width/height in pixels |
| `SMART_RETAIL_PRODUCT_MODEL` | `models/product_classifier.joblib` | Path to product classifier bundle |
| `SMART_RETAIL_SENTIMENT_MODEL` | `models/sentiment_classifier.joblib` | Path to sentiment classifier bundle |
| `SMART_RETAIL_FACE_DB` | `models/face_db.pkl` | Path to face database pickle |

---

## 5. Coding Standards

- **Formatting:** Adhere to PEP 8. Line length limit is 100 characters.
- **Type Annotations:** Full type hints on all function signatures (`from __future__ import annotations`).
- **Defensive Error Handling:** Never catch raw `Exception` without re-raising as an `HTTPException` with informative error context. Do not leak internal stack traces to HTTP clients.
- **Thread Safety:** Ensure shared state (e.g. counters) is guarded with `threading.Lock`.
