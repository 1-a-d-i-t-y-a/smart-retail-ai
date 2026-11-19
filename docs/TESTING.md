# Smart Retail AI Platform: Testing Documentation & Quality Assurance

**Total Tests:** 96  
**Passing Rate:** 100% (96 passed, 0 failed, 0 skipped)  
**Test Runner:** `pytest 9.1.1`  
**Execution Time:** ~5.2 seconds

---

## 1. Test Suite Structure

The automated test suite is located in `tests/` and structured into 9 modular suites:

| Test Module | Test Count | Scope & Behaviors Tested |
| :--- | :---: | :--- |
| `tests/test_config.py` | 2 | Default configuration integrity, environment variable overrides for image limits, dimension bounds, and model paths. |
| `tests/test_validation.py` | 11 | Payload validation: valid image matrix decoding, missing MIME type rejection, invalid MIME type rejection, empty byte buffer rejection, oversized byte buffer rejection, non-decodable byte corruption rejection, minimum dimension boundary check ($<10\text{px}$), maximum dimension boundary check ($>4096\text{px}$), bounded streaming chunk read within limit, bounded read exceeding limit (`PayloadTooLargeError`), empty file chunk read. |
| `tests/test_product.py` | 7 | Standard inference across product categories, invalid MIME handling (400), corrupt image payload (400), empty file upload (400), oversized upload handling (HTTP 413), undersized pixel dimensions (400), missing model artifact handling (503 Service Unavailable). |
| `tests/test_sentiment.py` | 7 | Positive sentiment review classification, negative review classification, neutral review classification, empty string rejection (422), pure whitespace string rejection (422), oversized text rejection ($>5000\text{ chars}$) (422), missing model artifact handling (503). |
| `tests/test_face.py` | 7 | Invalid MIME rejection (400), corrupt byte rejection (400), positive identity verification on registered customer (`person_01`), rejection of unregistered identity with `matched: false`, non-face distractor image reporting `No face detected`, oversized upload handling (HTTP 413), deterministic missing face database handling (503). |
| `tests/test_chatbot.py` | 16 | Parametrized tests covering all 9 operational retail support intents (`greeting`, `order_status`, `shipping`, `returns`, `refunds`, `store_info`, `products`, `payment`, `support`) + out-of-scope `fallback`, punctuation and casing normalization, empty message validation (422), whitespace validation (422). |
| `tests/test_dashboard.py` | 3 | Response schema verification (`total_requests`, `product_requests`, etc.), counter increment verification upon successful request, counter isolation verification (failed / 422 requests do not increment counters). |
| `tests/test_health.py` | 4 | Root redirect to Swagger UI (`/docs`), service health check (`/health`), readiness check when all models are present (`/health/ready` -> `ready`), readiness check when a model file is missing (`not_ready`). |
| `tests/test_packaging.py` | 39 | Packaging defense: 28 path classification tests verifying exclusion of `.git`, `__pycache__`, virtualenvs, credentials (`id_rsa`, `.pem`, `.key`), `.env` files, build output, and OS metadata; 7 regex secret scanner tests detecting GitHub PATs, OpenAI keys, AWS access keys, PEM blocks, password/token/secret assignments, and binary skip logic; 4 exclusion and secret pattern constant set integrity checks. |

---

## 2. Running Tests

```bash
# Execute entire test suite
pytest -v

# Run with fail-fast mode (stop at first error)
pytest -x

# Run only product and sentiment tests
pytest tests/test_product.py tests/test_sentiment.py -v

# Generate terminal coverage report
pytest --cov=app --cov-report=term-missing tests/
```

---

## 3. Fixtures Catalog (`tests/conftest.py`)

- `client`: Session-scoped FastAPI `TestClient` initialized with `app.main.app`.
- `sample_image_bytes`: Generates a valid in-memory $128 \times 128 \times 3$ JPEG image with encoded rectangle features.
- `invalid_image_bytes`: Supplies corrupted non-image raw bytes (`b"not_a_real_image_payload_binary_data"`) to verify parser error handling.

