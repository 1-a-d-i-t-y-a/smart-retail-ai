# Smart Retail AI Platform: API Reference

Base URL: `http://localhost:8000`  
Interactive OpenAPI Documentation: `/docs` (Swagger) | `/redoc` (ReDoc)

---

## 1. System Endpoints

### `GET /health`
Returns service availability and version metadata.
* **Response `200 OK`:**
  ```json
  {
    "status": "healthy",
    "service": "AI-Powered Smart Retail & Customer Intelligence Platform",
    "version": "1.1.0"
  }
  ```

### `GET /health/ready`
Checks readiness of all persistent machine learning model artifacts.
* **Response `200 OK`:**
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

---

## 2. Product Classification

### `POST /classify-product`
Classifies an uploaded product image into retail catalog categories.
* **Request:** Multipart form upload (`file: UploadFile`).
* **Response `200 OK`:**
  ```json
  {
    "product": "bottle",
    "confidence": 0.9421
  }
  ```
* **Error Responses:**
  * `400 Bad Request`: Invalid image format or corrupt bytes.
  * `503 Service Unavailable`: Model artifact not found.

---

## 3. Face Recognition

### `POST /recognize-face`
Verifies whether a customer face matches registered store profiles.
* **Request:** Multipart form upload (`file: UploadFile`).
* **Response `200 OK`:**
  ```json
  {
    "matched": true,
    "person": "person_01",
    "distance": 0.18,
    "similarity": 0.82,
    "bbox": [32, 28, 96, 96],
    "message": "Face matched"
  }
  ```

---

## 4. Sentiment Analysis

### `POST /analyze-sentiment`
Classifies customer review sentiment.
* **Request:** `application/json`
  ```json
  {
    "text": "The delivery was quick and the headphones sound incredible!"
  }
  ```
* **Response `200 OK`:**
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.9615
  }
  ```

---

## 5. Retail Chatbot

### `POST /chatbot`
Provides deterministic, domain-specific retail answers.
* **Request:** `application/json`
  ```json
  {
    "message": "Where is my order?"
  }
  ```
* **Response `200 OK`:**
  ```json
  {
    "intent": "order_status",
    "response": "Use the order ID with the order-status service to retrieve the latest shipment state."
  }
  ```

---

## 6. Dashboard & Telemetry

### `GET /dashboard/stats`
Retrieves in-process request counters and endpoint distributions.
* **Response `200 OK`:**
  ```json
  {
    "total_requests": 42,
    "product_requests": 10,
    "face_requests": 8,
    "sentiment_requests": 14,
    "chatbot_requests": 10
  }
  ```
