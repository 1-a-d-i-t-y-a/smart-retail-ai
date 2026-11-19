# Smart Retail AI Platform: Complete Reproducibility Guide

This document guarantees that external reviewers, evaluators, and engineers can replicate the entire machine learning pipeline, dataset creation, and automated test suite from zero on any modern machine.

---

## 1. Clean-Room Reproduction Workflow

Follow these steps to clean all generated artifacts and rebuild the system from scratch:

### Step 1: Wipe Existing Generated Data and Model Artifacts
```bash
# In project root:
# Remove serialized model weights
rm -rf models/*.joblib models/*.pkl

# Remove generated dataset directories
rm -rf data/training data/evaluation data/product_manifest.csv
```

### Step 2: Run Unified Build Pipeline
Execute the master build script:
```bash
python scripts/build_all_models.py
```

### Expected Output
```text
======================================================================
SMART RETAIL AI — UNIFIED ML PIPELINE (BUILD & EVALUATE)
======================================================================
[1/3] Generating Product Image Datasets...
  Generated 240 train, 60 test, and 20 challenge product images in data
[2/3] Generating Sentiment Text Datasets...
  Generated 283 train, 95 test, and 20 challenge sentiment samples.
[3/3] Generating Face Verification Evaluation Pairs...
  Generated face evaluation pairs in data/evaluation/face
============================================================
Dataset generation complete! All splits match documented artifacts.
============================================================

[1/3] Training Sentiment Analysis Model...
  Training on 283 balanced customer reviews...
  Saved sentiment model to models/sentiment_classifier.joblib
[2/3] Training Product Classification Model...
  Total training feature matrix shape: (240, 1764)
  Saved product classification model to models/product_classifier.joblib
[3/3] Building Face Verification Database...
  Saved face database to models/face_db.pkl with reference bbox [6, 8, 111, 111]
============================================================
All ML models trained and saved to models/ successfully!
============================================================

[1/3] Evaluating Product Classifier...
  Product Test Accuracy: 100.00% (60 samples)
  Product Challenge Accuracy: 100.00% (20 samples)
[2/3] Evaluating Sentiment Classifier...
  Sentiment Test Accuracy: 100.00% (95 samples)
  Sentiment Challenge Accuracy: 55.00% (20 samples)
[3/3] Evaluating Face Verification Service...
  Face Verification Accuracy: 100.00% (3 cases)
============================================================
Evaluation complete! Verified artifacts saved to artifacts/
============================================================
```

---

## 2. Verifying Automated Tests

Execute the full suite of 52 tests:
```bash
pytest -v
```

### Expected Result
```text
============================= 52 passed in 6.08s ==============================
```

---

## 3. Verifying Interactive Endpoints

Start the API server:
```bash
uvicorn app.main:app --port 8000
```

1. **Verify Health:**
   ```bash
   curl http://localhost:8000/health
   # Returns: {"status": "healthy", "service": "...", "version": "1.2.0"}
   ```

2. **Verify Readiness:**
   ```bash
   curl http://localhost:8000/health/ready
   # Returns: {"status": "ready", "checks": {"product_model": true, "sentiment_model": true, "face_database": true}}
   ```

3. **Verify Product Classification:**
   ```bash
   curl -X POST http://localhost:8000/classify-product -F "file=@data/evaluation/product/mug/mug_test_000.png;type=image/png"
   # Returns: {"product": "mug", "confidence": 0.99...}
   ```

4. **Verify Sentiment Analysis:**
   ```bash
   curl -X POST http://localhost:8000/analyze-sentiment -H "Content-Type: application/json" -d '{"text": "Fast shipping, fantastic service!"}'
   # Returns: {"sentiment": "positive", "confidence": 0.96...}
   ```

5. **Verify Retail Chatbot:**
   ```bash
   curl -X POST http://localhost:8000/chatbot -H "Content-Type: application/json" -d '{"message": "What is your refund policy?"}'
   # Returns: {"intent": "refunds", "response": "Refunds are credited to the original payment method..."}
   ```
