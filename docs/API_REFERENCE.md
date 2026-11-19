# Smart Retail AI Platform: REST API Reference

**Base URL:** `http://localhost:8000`  
**Interactive Swagger UI:** `http://localhost:8000/docs`  
**ReDoc Specification:** `http://localhost:8000/redoc`  
**OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## 1. System Endpoints

### 1.1 Root Redirect
- **Route:** `GET /`
- **Description:** Automatically redirects web browsers to the interactive Swagger UI (`/docs`).
- **Response:** `307 Temporary Redirect` -> `/docs`

---

### 1.2 Health Check
- **Route:** `GET /health`
- **Summary:** Returns basic service uptime status and application metadata.
- **Request:**
  ```bash
  curl -s http://localhost:8000/health
  ```
- **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "AI-Powered Smart Retail & Customer Intelligence Platform",
    "version": "1.2.0"
  }
  ```

---

### 1.3 Model Readiness Check
- **Route:** `GET /health/ready`
- **Summary:** Verifies the physical presence of required machine learning artifacts on the host filesystem.
- **Request:**
  ```bash
  curl -s http://localhost:8000/health/ready
  ```
- **Response (200 OK — Ready):**
  ```json
  {
    "status": "ready",
    "checks": {
      "product_model": true,
      "sentiment_model": true,
      "face_database": true
    }
  }
  ```
- **Response (200 OK — Degraded):**
  ```json
  {
    "status": "not_ready",
    "checks": {
      "product_model": false,
      "sentiment_model": true,
      "face_database": true
    }
  }
  ```

---

## 2. Product Classification

### 2.1 Classify Product Image
- **Route:** `POST /classify-product`
- **Description:** Analyzes an uploaded product image using HOG feature descriptors and an RBF-kernel SVM.
- **Headers:** `Content-Type: multipart/form-data`
- **Parameters:**
  - `file`: Binary image file (`JPEG`, `PNG`, `WebP`).
  - Limits: File size $\le 5\text{ MB}$, dimensions between $10\times 10$ and $4096\times 4096$ pixels.
- **Request:**
  ```bash
  curl -X POST http://localhost:8000/classify-product \
       -F "file=@data/evaluation/product/bottle/bottle_test_000.png;type=image/png"
  ```
- **Response (200 OK):**
  ```json
  {
    "product": "bottle",
    "confidence": 0.9854
  }
  ```
- **Error Responses:**
  - `400 Bad Request`: Invalid image format, corrupted payload, or dimensions out of bounds.
    ```json
    {
      "detail": "Could not decode image: file may be corrupted or in an unsupported format."
    }
    ```
  - `503 Service Unavailable`: Product classification model file missing.
    ```json
    {
      "detail": "Product classification model is unavailable. Please ensure models are built."
    }
    ```

---

## 3. Customer Face Verification

### 3.1 Recognize Face
- **Route:** `POST /recognize-face`
- **Description:** Scans for human faces via Haar Cascade and computes structural similarity (SSIM) and Pearson correlation against the registered customer database.
- **Headers:** `Content-Type: multipart/form-data`
- **Parameters:**
  - `file`: Binary image file (`JPEG`, `PNG`, `WebP`).
- **Request:**
  ```bash
  curl -X POST http://localhost:8000/recognize-face \
       -F "file=@data/training/face/person_01_reference.png;type=image/png"
  ```
- **Response (200 OK — Matched):**
  ```json
  {
    "matched": true,
    "person": "person_01",
    "distance": 0.0,
    "similarity": 1.0,
    "bbox": [6, 8, 111, 111],
    "message": "Face matched"
  }
  ```
- **Response (200 OK — No Registered Match):**
  ```json
  {
    "matched": false,
    "person": null,
    "distance": 0.7241,
    "similarity": 0.2759,
    "bbox": [284, 125, 127, 127],
    "message": "Face detected but no registered match"
  }
  ```
- **Response (200 OK — No Face Detected):**
  ```json
  {
    "matched": false,
    "person": null,
    "distance": null,
    "similarity": null,
    "bbox": null,
    "message": "No face detected in query image."
  }
  ```
- **Error Responses:**
  - `400 Bad Request`: Corrupted image or invalid MIME type.
  - `503 Service Unavailable`: Face database file not found.

---

## 4. Customer Sentiment Analysis

### 4.1 Analyze Sentiment
- **Route:** `POST /analyze-sentiment`
- **Description:** Evaluates customer review text using TF-IDF tokenization and Logistic Regression.
- **Headers:** `Content-Type: application/json`
- **Body:**
  ```json
  {
    "text": "The delivery arrived early and the packaging was immaculate!"
  }
  ```
- **Validation:** String length must be between 1 and 5,000 characters; pure whitespace strings are rejected.
- **Request:**
  ```bash
  curl -X POST http://localhost:8000/analyze-sentiment \
       -H "Content-Type: application/json" \
       -d '{"text": "The delivery arrived early and the packaging was immaculate!"}'
  ```
- **Response (200 OK):**
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.9634
  }
  ```
- **Error Responses:**
  - `422 Unprocessable Entity`: Missing body, empty text, or whitespace-only text.
  - `503 Service Unavailable`: Sentiment classification model missing.

---

## 5. Retail Chatbot

### 5.1 Query Chatbot Assistant
- **Route:** `POST /chatbot`
- **Description:** Dispatches customer queries to 9 retail operational intents (`greeting`, `order_status`, `shipping`, `returns`, `refunds`, `store_info`, `products`, `payment`, `support`) or a safe `fallback`.
- **Headers:** `Content-Type: application/json`
- **Body:**
  ```json
  {
    "message": "How do I return this shoe?"
  }
  ```
- **Request:**
  ```bash
  curl -X POST http://localhost:8000/chatbot \
       -H "Content-Type: application/json" \
       -d '{"message": "How do I return this shoe?"}'
  ```
- **Response (200 OK):**
  ```json
  {
    "intent": "returns",
    "response": "Return or exchange requests should be raised through the support workflow with your order ID within 30 days."
  }
  ```
- **Error Responses:**
  - `422 Unprocessable Entity`: Blank message or message exceeding 2,000 characters.

---

## 6. Analytics Dashboard

### 6.1 System Usage Metrics
- **Route:** `GET /dashboard/stats`
- **Description:** Returns thread-safe in-process execution statistics across all endpoints. Counters increment exclusively on successful inference runs.
- **Request:**
  ```bash
  curl -s http://localhost:8000/dashboard/stats
  ```
- **Response (200 OK):**
  ```json
  {
    "total_requests": 42,
    "product_requests": 15,
    "face_requests": 8,
    "sentiment_requests": 11,
    "chatbot_requests": 8
  }
  ```
