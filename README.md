[README.md](https://github.com/user-attachments/files/32665425/README.md)
# AI-Powered Smart Retail & Customer Intelligence Platform

A modular, reproducible AI/ML retail platform combining **product image classification, face verification, sentiment analysis, a deterministic retail chatbot, and a FastAPI gateway**.

> **Important:** This repository is a reconstructed/re-implemented academic project. Its generated datasets, metrics, screenshots, and evaluation artifacts are project-development evidence, not original historical internship telemetry.

## Features

- Product classification using HOG + RBF SVM
- Lightweight face verification using OpenCV and similarity matching
- Retail sentiment analysis using TF-IDF + Logistic Regression
- Deterministic retail chatbot with local knowledge base
- FastAPI REST endpoints
- Pydantic validation and controlled error handling
- Health and readiness checks
- Request statistics and structured logging
- Swagger/OpenAPI documentation
- Automated regression tests
- Controlled and challenge-set model evaluation
- Docker/Docker Compose configuration
- GitHub Actions CI

## Architecture

```text
Client
  |
  v
FastAPI Gateway
  |
  +--> Product Service --> HOG --> RBF SVM
  |
  +--> Face Service ------> OpenCV --> Similarity Matcher
  |
  +--> Sentiment Service -> TF-IDF -> Logistic Regression
  |
  +--> Chatbot Service ---> Intent Router -> Retail Knowledge Base
  |
  +--> Dashboard / Health / Readiness
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| GET | `/health/ready` | Model/artifact readiness |
| POST | `/classify-product` | Product classification |
| POST | `/recognize-face` | Face verification |
| POST | `/analyze-sentiment` | Sentiment analysis |
| POST | `/chatbot` | Retail chatbot |
| GET | `/dashboard/stats` | Request statistics |

Interactive API documentation is available at `/docs` after starting the application.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/<YOUR_USERNAME>/smart-retail-ai.git
cd smart-retail-ai
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Build/rebuild model artifacts

```bash
python scripts/build_all_models.py
```

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Testing

Run the complete test suite:

```bash
python -m pytest -q
```

The reconstructed project currently contains **39 automated tests**, with the final local run reporting **39 passed**.

## Evaluation

Controlled and challenge evaluation are intentionally separated.

| Component | Controlled | Challenge |
|---|---:|---:|
| Product Classification | 100% CV | 60.0% accuracy |
| Face Verification | 10/10 controlled benchmark | Not sufficient for generalization |
| Sentiment Analysis | 100% held-out | 38.9% accuracy |
| Chatbot | 9/9 scenarios | Functional |
| End-to-End | 5/5 workflows | PASS |

The challenge results are included to avoid presenting perfect controlled scores as evidence of unrestricted real-world generalization.

## Docker

Docker configuration is included in `Dockerfile` and `docker-compose.yml`.

The Docker configuration was prepared and tested structurally, but an actual container runtime was not executed in the reconstruction environment because the Docker CLI was unavailable. Do not interpret the repository as evidence of a completed container deployment.

## Repository Structure

```text
smart-retail-ai/
├── app/                 # FastAPI application and AI services
├── data/                # Reconstructed local datasets
├── models/              # Persisted model artifacts
├── artifacts/           # Evaluation and API evidence
├── docs/                # Technical documentation
├── scripts/             # Training, evaluation and verification scripts
├── tests/               # Automated tests
├── .github/workflows/   # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## Documentation

- [Complete AI/ML Project Documentation](docs/PROJECT_DOCUMENTATION.md)
- [API Documentation](docs/api_documentation.md)
- [Architecture](docs/architecture.svg)
- [Chatbot](docs/chatbot.md)
- [Face Verification](docs/face_recognition.md)
- [Sentiment Analysis](docs/sentiment_analysis.md)

## Selected Evidence

### Product Classification API

![Product Classification API](docs/screenshots/01-product-classification-api.png)

### Sentiment Analysis API

![Sentiment Analysis API](docs/screenshots/02-sentiment-analysis-api.png)

### Automated Tests

![Pytest](docs/screenshots/03-pytest-39-passed.png)

### Face Verification

![Face Verification](docs/screenshots/04-face-verification.png)

## Academic / Provenance Note

This project was reconstructed as a reproducible technical implementation. Generated datasets, screenshots, metrics and evaluation results should be described as reconstructed project evidence in academic reporting unless independently supported by original records.

## License

MIT License. See [LICENSE](LICENSE).
