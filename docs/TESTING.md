# Smart Retail AI Platform: Testing Documentation & Quality Assurance

**Total Tests:** 129
**Passing Rate:** 100% (129 passed, 0 failed, 0 skipped)
**Test Runner:** `pytest 9.1.1`  
**Execution Time:** ~5.8 seconds

---

## 1. Test Suite Structure

The automated test suite is located in `tests/` and structured into 9 modular suites:

| Test Module | Test Count | Scope & Behaviors Tested |
| :--- | :---: | :--- |
| `tests/test_config.py` | 2 | Default configuration integrity, environment variable overrides for image limits, dimension bounds, and model paths. |
| `tests/test_validation.py` | 20 | Strict Content-Type allowlist normalization (`image/jpeg`, `image/png`, `image/webp`, `image/bmp`), rejection of unauthorized types (`image/svg+xml`, `text/plain`), MIME/format mismatch detection, empty byte buffer rejection, oversized payload rejection, corrupt payload rejection, minimum dimension boundary check ($<10\text{px}$), maximum dimension boundary check ($>4096\text{px}$), Pillow header decompression-bomb traps, bounded streaming read at exact limit, bounded read exceeding limit (`PayloadTooLargeError`), immediate read cessation verification (stopping at limit + 1 byte), empty file chunk reads, and invalid `max_bytes` configuration validation. |
| `tests/test_product.py` | 7 | Standard inference across product categories (`bottle`, `mug`, `headphones`, `shoe`), invalid MIME handling (400), corrupt image payload (400), empty file upload (400), oversized upload handling (HTTP 413), undersized pixel dimensions (400), missing model artifact handling (503 Service Unavailable). |
| `tests/test_sentiment.py` | 7 | Positive sentiment review classification, negative review classification, neutral review classification, empty string rejection (422), pure whitespace string rejection (422), oversized text rejection ($>5000\text{ chars}$) (422), missing model artifact handling (503). |
| `tests/test_face.py` | 7 | Invalid MIME rejection (400), corrupt byte rejection (400), positive identity verification on registered customer (`person_01`), rejection of unregistered identity with `matched: false`, non-face distractor image reporting `No face detected`, oversized upload handling (HTTP 413), deterministic missing face database handling (503). |
| `tests/test_chatbot.py` | 16 | Parametrized tests covering all 9 operational retail support intents (`greeting`, `order_status`, `shipping`, `returns`, `refunds`, `store_info`, `products`, `payment`, `support`) + out-of-scope `fallback`, punctuation and casing normalization, empty message validation (422), whitespace validation (422). |
| `tests/test_dashboard.py` | 3 | Response schema verification (`total_requests`, `product_requests`, etc.), counter increment verification upon successful request, counter isolation verification (failed / 422 requests do not increment counters). |
| `tests/test_health.py` | 4 | Root redirect to Swagger UI (`/docs`), service health check (`/health`), readiness check when all models are present (`/health/ready` -> `ready`), readiness check when a model file is missing (`not_ready`). |
| `tests/test_packaging.py` | 63 | Packaging defense: 41 path classification tests verifying exclusion of `.git`, `__pycache__`, virtualenvs, credentials (`id_rsa`, `.pem`, `.key`), `.env` files, build output, and OS metadata, while ensuring `.env.example` is retained; 3 scannability tests; 10 regex secret scanner tests detecting GitHub fine-grained PATs (`github_pat_`), classic PATs (`ghp_`), OAuth tokens (`gho_`), OpenAI keys, AWS access keys, PEM blocks, JSON/key-value password/token assignments, and safe placeholder filtering; 6 archive path and traversal validation tests (safe paths, absolute paths, Windows drive letters, UNC paths, `../` traversal, backslash traversal, symlinks); 3 packaging constant set integrity tests. |

---

## 2. Running Tests

```bash
# Execute entire test suite
pytest -v

# Run with fail-fast mode (stop at first error)
pytest -x

# Run only validation and packaging security tests
pytest tests/test_validation.py tests/test_packaging.py -v

# Generate terminal coverage report
pytest --cov=app --cov-report=term-missing tests/
```

---

## 3. Fixtures Catalog (`tests/conftest.py`)

- `client`: Session-scoped FastAPI `TestClient` initialized with `app.main.app`.
- `sample_image_bytes`: Generates a valid in-memory $128 \times 128 \times 3$ JPEG image with encoded rectangle features.
- `invalid_image_bytes`: Supplies corrupted non-image raw bytes (`b"not_a_real_image_payload_binary_data"`) to verify parser error handling.
