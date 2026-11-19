# Smart Retail AI Platform: Model Evaluation Report

**Evaluation Date:** November 19, 2026  
**Artifact Generation:** Deterministic via `scripts/evaluate_models.py`  
**Random Seed:** 42  
**Dataset Origin:** Synthetic retail benchmarks matching documented splits.

---

## 1. Executive Summary

This evaluation measures the performance of the three machine learning pipelines powering Smart Retail AI:
1. **Product Image Classifier:** HOG Feature Extractor + RBF-kernel SVM.
2. **Customer Sentiment Classifier:** TF-IDF Vectorizer + Multinomial Logistic Regression.
3. **Face Verification System:** OpenCV Haar Cascade Detector + SSIM & Pearson Correlation Matcher.

All metrics below are 100% reproducible and generated locally without external cloud telemetry or third-party paid APIs.

---

## 2. Product Image Classification Evaluation

### 2.1 Benchmark Specification
- **Architecture:** Pipeline(`HOGDescriptor` [1764 features] -> `SVC(kernel="rbf", C=5.0, probability=True)`)
- **Training Set:** 240 synthetic images (60 per class: `bottle`, `mug`, `headphones`, `shoe`)
- **Test Set:** 60 held-out images (15 per class)
- **Challenge Set:** 20 images with extreme sensor noise, illumination variance, and geometric shifts

### 2.2 Test Set Performance
$$\text{Overall Accuracy} = 100.00\% \quad (60/60)$$

> **Evaluation Notice:** Achieved 100% accuracy on the controlled synthetic held-out benchmark (60/60). This benchmark demonstrates reproducible behavior on generated data and should not be interpreted as real-world retail generalization.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Bottle** | 1.000 | 1.000 | 1.000 | 15 |
| **Mug** | 1.000 | 1.000 | 1.000 | 15 |
| **Headphones** | 1.000 | 1.000 | 1.000 | 15 |
| **Shoe** | 1.000 | 1.000 | 1.000 | 15 |
| **Macro Average** | 1.000 | 1.000 | 1.000 | 60 |
| **Weighted Average** | 1.000 | 1.000 | 1.000 | 60 |

### 2.3 Confusion Matrix
```text
                Predicted Bottle  Predicted Mug  Predicted Headphones  Predicted Shoe
Actual Bottle          15               0                 0                  0
Actual Mug              0              15                 0                  0
Actual Headphones       0               0                15                  0
Actual Shoe             0               0                 0                 15
```

### 2.4 Challenge Set Results
- **Evaluated Samples:** 20 noisy / off-center images (5 per class).
- **Challenge Accuracy:** $100.00\%$ ($20/20$).
- **Observation:** HOG feature extraction displays robust orientation invariance against localized pixel noise and translation within standard bounding frames.

---

## 3. Customer Sentiment Analysis Evaluation

### 3.1 Benchmark Specification
- **Architecture:** Pipeline(`TfidfVectorizer(ngram_range=(1,2), min_df=1)` -> `LogisticRegression(C=1.0, max_iter=300)`)
- **Training Set:** 283 balanced retail reviews (94 negative, 95 neutral, 94 positive)
- **Test Set:** 95 held-out reviews (31 negative, 32 neutral, 32 positive)
- **Challenge Set:** 20 highly nuanced reviews with idioms, negations, mixed polarities, and sarcasm

### 3.2 Test Set Performance
$$\text{Overall Accuracy} = 100.00\% \quad (95/95)$$

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 1.000 | 1.000 | 1.000 | 31 |
| **Neutral** | 1.000 | 1.000 | 1.000 | 32 |
| **Positive** | 1.000 | 1.000 | 1.000 | 32 |
| **Macro Average** | 1.000 | 1.000 | 1.000 | 95 |
| **Weighted Average** | 1.000 | 1.000 | 1.000 | 95 |

### 3.3 Confusion Matrix
```text
                 Predicted Negative  Predicted Neutral  Predicted Positive
Actual Negative          31                  0                  0
Actual Neutral            0                 32                  0
Actual Positive           0                  0                 32
```

### 3.4 Challenge Set Results (Honest Negative Case Analysis)
- **Evaluated Samples:** 20 complex review sentences.
- **Challenge Accuracy:** $55.00\%$ ($11/20$).
- **Insight:** Classical n-gram models struggle with complex linguistic phenomena such as:
  - *Mixed sentiment:* "The box was beautiful, shame the actual bottle leaked everywhere." (Model predicts positive due to "beautiful", true label is negative).
  - *Double negatives:* "Not bad at all for the price, surprisingly good sound." (Correctly predicted positive).
  - *Subtle sarcasm:* "Expected something revolutionary, but it is purely standard." (Predicted positive, true label neutral).
- **Conclusion:** While baseline TF-IDF is lightning-fast (<1ms per inference) and 100% accurate on straightforward feedback, production deployments with conversational user feedback would benefit from transformer embeddings (e.g. RoBERTa or DistilBERT).

---

## 4. Customer Face Verification Evaluation

### 4.1 Benchmark Specification
- **Detector:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- **Normalizer:** $160 \times 160$ grayscale resize with Histogram Equalization
- **Metric:** $0.70 \times \text{SSIM} + 0.30 \times \text{Normalized Pearson Correlation}$
- **Threshold:** $\tau = 0.50$

### 4.2 Test Pair Results
$$\text{Verification Accuracy} = 100.00\% \quad (3/3 \text{ benchmark test scenarios})$$

| Scenario | Input Image | Expected | Predicted | Similarity | Distance | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Registered Identity** | `person_01_reference.png` | Match | Match | 1.0000 | 0.0000 | Correct |
| **Positive Query** | `positive_query.png` (illumination shifted) | Match | Match | 0.9953 | 0.0047 | Correct |
| **Unregistered Identity** | `negative_other_face.png` | No Match | No Match | 0.2759 | 0.7241 | Correct |
| **Non-Face Distractor** _(functional test — not in accuracy denominator)_ | `no_face_distractor.png` | No Match | No Match | N/A | N/A | ✅ Code path verified |

### 4.3 Engineering Scope & Limitations
- **Educational / Demo Purpose:** The face verification pipeline is designed to demonstrate computer vision image processing fundamentals.
- **Not Biometric Grade:** It does not use deep representation networks (such as ArcFace or FaceNet) and lacks liveness detection, 3D anti-spoofing, and pose invariance.

