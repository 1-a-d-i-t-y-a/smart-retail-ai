# Model Evaluation & Benchmark Methodology

> **Academic Provenance Notice:** This platform reconstructs a retail intelligence system using classical, offline-capable machine learning algorithms. The benchmarks below represent controlled validation evaluations on synthetic and benchmark test sets, rather than live production retail telemetry.

---

## 1. Product Image Classifier (HOG + RBF SVM)

* **Features:** 128x128 HOG descriptor with 9 orientation bins.
* **Classifier:** Support Vector Classifier with Radial Basis Function kernel ($C=5.0$, probability calibrated).
* **Classes:** `bottle`, `mug`, `headphones`, `shoe`.
* **Controlled Evaluation Results:**
  * Controlled Test Samples: 60
  * Classification Accuracy: 100% on synthetic geometric benchmarks.
  * *Real-World Generalization Note:* Real-world product photography with complex backgrounds, occlusion, and varied lighting will have higher variance; data augmentation and transfer learning (e.g., MobileNetV3) are recommended for production deployments with complex backgrounds.

---

## 2. Sentiment Classifier (TF-IDF + Logistic Regression)

* **Features:** Unigram and bigram TF-IDF vectorization.
* **Classifier:** Multinomial Logistic Regression ($C=1.0$, L2 penalty).
* **Classes:** `positive`, `neutral`, `negative`.
* **Controlled Evaluation Results:**
  * Controlled Test Samples: 95
  * Macro F1-Score: 1.0 on balanced benchmark evaluation cases.
  * *Real-World Generalization Note:* Slang, sarcasm, and multilingual queries represent known boundary cases where modern transformer models (e.g., DistilBERT) offer superior generalization.

---

## 3. Face Verification (Haar Cascade + SSIM/Correlation)

* **Matching Algorithm:**
  $$\text{Combined Score} = 0.70 \times \text{SSIM} + 0.30 \times \text{NormCorr}$$
* **Threshold:** 0.52 for positive verification.
* **Evaluation:** Tested across 10 positive reference captures and 10 negative distractor captures.
* **Controlled Accuracy:** 100% true acceptance rate on controlled frontal facial captures with consistent illumination.
