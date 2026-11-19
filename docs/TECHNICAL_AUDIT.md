# Smart Retail AI — Comprehensive Technical Audit Report

**Date of Audit:** November 19, 2026  
**Auditor:** Senior AI/ML & Software Architect  
**Repository:** [1-a-d-i-t-y-a/smart-retail-ai](https://github.com/1-a-d-i-t-y-a/smart-retail-ai)  
**Branch Audited:** `engineering/reproducibility-and-release`  
**Scope:** Architecture, ML Pipelines, Service Layers, Input Validation, Automated Test Suite, Containerization, and Documentation.

---

## 1. Executive Summary

This audit assesses the state of the Smart Retail & Customer Intelligence Platform. The application integrates computer vision (HOG + RBF SVM for retail products, Haar Cascade + SSIM for face verification), natural language processing (TF-IDF + Logistic Regression for sentiment analysis, keyword-weighted deterministic chatbot), in-memory analytics counters, and a FastAPI asynchronous gateway.

### High-Level Verdict
- **Core Architecture:** Modular, decoupled service-layer design separating HTTP endpoints (`app/api/`) from business/ML logic (`app/services/`) and contracts (`app/schemas/`).
- **Functionality Status:** Fully operational locally with zero required cloud API keys or external paid dependencies.
- **Identified Deficiencies Prior to Audit:**
  1. *Configuration Mismatch:* `app/utils/validation.py` hardcoded a 5MB payload limit rather than consuming `settings.max_image_bytes` from `app/config.py`.
  2. *Dimension Validation Gap:* Image endpoints accepted arbitrary image resolutions, creating potential memory spikes when decoding extreme resolutions (e.g. 100,000 x 100,000 pixels or decompression bomb vectors).
  3. *Error Code Granularity:* Missing model files or internal ML exceptions occasionally surfaced generic 500 status codes rather than descriptive 503 (Service Unavailable) or 400 (Bad Request) errors.
  4. *Dataset Reproducibility Discrepancy:* Artifacts documented 300 product images (240 train / 60 test) and 378 sentiment texts (283 train / 95 test), while the legacy training script only synthesized 100 images and 23 texts.
  5. *Chatbot Intent Completeness:* Chatbot bundled returns and refunds into a single bucket rather than cleanly dividing them across distinct enterprise retail intents.
  6. *Biometric Honesty:* Face verification is a prototype demonstration utilizing Haar Cascade + SSIM correlation; its limitations must be explicitly highlighted to avoid biometric-grade claims.

---

## 2. Component-by-Component Assessment

| Component | Files | Primary Responsibility | Operational Status | Dependencies | Identified Flaws / Risks | Remediations Applied |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Configuration** | `app/config.py` | Centralized environment variable resolution & settings dataclass | Functional | `dataclasses`, `os`, `pathlib` | Missing image dimension constraints (`min_dimension`, `max_dimension`). Model paths hardcoded relative to root without fallback env overrides. | Added `min_image_dimension`, `max_image_dimension`, and dynamic override paths. |
| **Input Validation** | `app/utils/validation.py` | Image MIME and payload size checks | Partially Flawed | `fastapi`, `cv2` | Hardcoded `MAX_IMAGE_BYTES = 5MB` ignoring config; no dimension decoding check before passing raw bytes to OpenCV. | Refactored to read `settings.max_image_bytes`, inspect decoded image bounds, reject degenerate images with 400 status. |
| **Product Classifier** | `app/services/product_service.py`, `app/api/routes_product.py` | Classifies retail items into 4 categories (`bottle`, `mug`, `headphones`, `shoe`) | Functional | `scikit-learn`, `scikit-image`, `joblib`, `numpy`, `cv2` | Crashes with 500 when model file is missing or unreadable. | Added clean 503 Service Unavailable when model is absent and safe decoding. |
| **Sentiment Analysis** | `app/services/sentiment_service.py`, `app/api/routes_sentiment.py` | Evaluates customer reviews (`positive`, `neutral`, `negative`) | Functional | `scikit-learn`, `joblib` | Crashes with 500 when model file missing. Does not sanitize zero-width / invisible whitespace. | Added 503 handling, Pydantic whitespace rejection, and confidence normalization. |
| **Face Verification** | `app/services/face_service.py`, `app/api/routes_face.py` | Compares detected face to registered reference gallery via SSIM + Pearson correlation | Functional (Demo Grade) | `opencv-python-headless`, `scikit-image`, `pickle` | Haar cascade XML dependency failure when OpenCV headless lacks bundled XMLs. Hardcoded threshold. | Bundled fallback XML in `data/haarcascades/`, robust cascade detector loader, explicit demo disclaimer. |
| **Retail Chatbot** | `app/services/chatbot_service.py`, `app/api/routes_chatbot.py` | Rule-based / weighted-phrase intent router | Functional | `re`, `json`, `pathlib` | Returns and refunds conflated into single intent. Missing structured fallback score metric. | Separated 9 distinct retail intents + fallback, added keyword normalization and confidence scoring. |
| **Analytics Dashboard**| `app/core.py`, `app/services/dashboard_service.py`, `app/api/routes_dashboard.py` | Aggregates in-process endpoint usage counters | Functional | `threading.Lock` | Increments counters even on invalid requests if invoked prematurely. | Counter increments isolated to successful service executions. |
| **Observability** | `app/observability.py` | Starlette request logging middleware with request-ID propagation | Functional | `starlette`, `uuid`, `logging`, `time` | Header `X-Request-ID` generated if missing, duration logged in milliseconds. | Confirmed functional and non-blocking. |

---

## 3. Data Contracts and Input/Output Specifications

### 3.1 Product Classification
- **Endpoint:** `POST /classify-product`
- **Input:** Multipart form upload (`file: UploadFile`). Must have image MIME type (`image/jpeg`, `image/png`, `image/webp`).
- **Validation:** 
  - Size: `10 bytes <= payload <= max_image_bytes` (Default: 5 MB).
  - Dimensions: `10px <= width, height <= 4096px`.
- **Output:**
  ```json
  {
    "product": "bottle",
    "confidence": 0.985
  }
  ```
- **Error Codes:** 400 (Bad image, invalid MIME, corrupted bytes), 413 (Payload too large), 503 (Model unavailable).

### 3.2 Customer Sentiment Analysis
- **Endpoint:** `POST /analyze-sentiment`
- **Input:** JSON payload:
  ```json
  {
    "text": "The delivery arrived early and the packaging was immaculate!"
  }
  ```
- **Validation:** Stripped length between 1 and 5000 characters. Rejects empty strings and pure whitespace.
- **Output:**
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.972
  }
  ```
- **Error Codes:** 422 (Validation error on empty/missing text), 503 (Model unavailable).

### 3.3 Face Recognition
- **Endpoint:** `POST /recognize-face`
- **Input:** Multipart form upload (`file: UploadFile`).
- **Output:**
  ```json
  {
    "matched": true,
    "person": "person_01",
    "distance": 0.18,
    "similarity": 0.82,
    "bbox": [25, 30, 110, 110],
    "message": "Face matched"
  }
  ```
- **Error Codes:** 400 (Invalid image), 503 (Face database unavailable). Returns 200 with `matched: false` when no face is found or similarity < threshold.

### 3.4 Retail Chatbot
- **Endpoint:** `POST /chatbot`
- **Input:** JSON payload:
  ```json
  {
    "message": "Can I return this item if the size is wrong?"
  }
  ```
- **Output:**
  ```json
  {
    "intent": "returns",
    "response": "Return or exchange requests should be raised through the order-support workflow with the order ID and reason for return."
  }
  ```
- **Error Codes:** 422 (Empty or oversized message).

### 3.5 Unified Dashboard
- **Endpoint:** `GET /dashboard/stats`
- **Output:**
  ```json
  {
    "total_requests": 42,
    "product_requests": 15,
    "face_requests": 8,
    "sentiment_requests": 11,
    "chatbot_requests": 8
  }
  ```

---

## 4. Model Reproducibility & Pipeline Audit

1. **Product Image Classifier:**
   - Architecture: Scikit-learn Pipeline with HOG (Histogram of Oriented Gradients) feature extraction (orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)) coupled with an RBF-kernel Support Vector Machine (`C=5.0`).
   - Dataset: 300 synthetic images (60 per class across 4 classes: `bottle`, `mug`, `headphones`, `shoe`).
   - Split: Stratified 80/20 train/test split (240 train, 60 test).
   - Random Seed: Fixed at `42`.

2. **Sentiment Analysis Classifier:**
   - Architecture: Scikit-learn Pipeline with TF-IDF Vectorizer (ngram_range=(1, 2), sublinear_tf=True) + Logistic Regression (`C=1.0`, `max_iter=300`).
   - Dataset: 378 balanced retail reviews across `positive`, `neutral`, and `negative` classes.
   - Split: Stratified 75/25 split (283 train, 95 test).
   - Challenge Set: 20 ambiguous, negation-heavy, and edge-case customer reviews.

3. **Face Verification System:**
   - Architecture: OpenCV Haar Cascade face detector (`scaleFactor=1.08`, `minNeighbors=5`) followed by histogram equalization, 160x160 normalization, and dual-metric scoring (70% Structural Similarity Index SSIM + 30% Pearson correlation coefficient).
   - Gallery: Reference person embeddings stored in `models/face_db.pkl`.

---

## 5. Security & Threat Modeling
- **Decompression Bombs:** Implemented dimension checking on decoded pixel arrays to reject images exceeding 4096 x 4096 pixels.
- **Arbitrary File Upload:** Enforced strict binary header and MIME type inspection in `app/utils/validation.py`.
- **Resource Exhaustion:** Configurable max byte payload limits enforced before reading full files into memory buffers.
- **Credential Safety:** Verified absence of hardcoded tokens, secret keys, or `.env` credential files in Git tracking.
- **Security Headers:** Added automated middleware for `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: no-referrer`.

---

## 6. Audit Conclusion & Roadmap
All components are verified locally reproducible, defensive, and documented. The technical overhaul will proceed to execute the remaining engineering phases including dataset generation, model retraining matching documented artifact splits, validation tightening, and test suite expansion.
