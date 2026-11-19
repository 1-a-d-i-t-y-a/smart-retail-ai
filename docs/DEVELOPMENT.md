# Development & Contributing Guide

## Prerequisites
- Python 3.11 or 3.12
- Git
- (Optional) Docker & Docker Compose

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/1-a-d-i-t-y-a/smart-retail-ai.git
   cd smart-retail-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Build baseline model weights:**
   ```bash
   python scripts/train_baseline_models.py
   ```

5. **Run the automated test suite:**
   ```bash
   pytest tests/ -v
   ```

6. **Start the development server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Navigate to `http://localhost:8000/docs` to view Swagger UI.

---

## Docker Execution

Build and start the containerized service:
```bash
docker compose up --build
```
Healthcheck will automatically probe `http://127.0.0.1:8000/health/ready`.
