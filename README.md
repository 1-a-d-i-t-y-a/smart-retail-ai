# AI Smart Retail & Customer Intelligence Platform

An end-to-end AI system for modern retail combining **Computer Vision**, **Deep Learning**,
**Natural Language Processing**, and a unified **FastAPI** microservice — delivered as a
5-day project across Google Colab notebooks and this GitHub repository.

---

## Modules

| Day | Module | Notebook | Description |
|-----|--------|----------|-------------|
| 1 | Setup & Data | `notebooks/01_day1_setup_and_data.ipynb` | GitHub + Drive setup, folder tree, dataset download |
| 2 | Face / Customer ID | `notebooks/02_day2_cv_utils.ipynb` | OpenCV image utils + Haar-cascade face recognition |
| 3 | Product Classifier | `notebooks/03_day3_product_classifier.ipynb` | MobileNetV2 transfer-learning CNN |
| 4 | Sentiment + Chatbot | `notebooks/04_day4_sentiment_chatbot.ipynb` | TF-IDF sentiment + retrieval FAQ bot |
| 5 | Unified FastAPI + ngrok | `notebooks/05_day5_fastapi_deployment.ipynb` | Serve all four models behind one API |

---

## Repository Layout

```
Major_Project/
├── data/                     # Sample datasets (products, faces, reviews)
├── models/                   # Trained model artefacts (created after training)
├── notebooks/                # 5 sequential Google Colab notebooks
├── src/                      # Reusable Python modules
│   ├── image_utils.py        # Resize / normalise / augment / grayscale
│   ├── face_utils.py         # Haar-cascade detection + embedding store
│   ├── product_classifier.py # MobileNetV2 builder + train helpers
│   ├── sentiment_model.py    # TF-IDF + Logistic-Regression sentiment
│   ├── chatbot.py            # Retrieval-based retail FAQ bot
│   └── api.py                # Unified FastAPI application
├── tests/                    # Pytest smoke tests
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

```bash
git clone <your-repo-url>
cd Major_Project
pip install -r requirements.txt

# Run the API locally
uvicorn src.api:app --reload --port 8000
# open http://localhost:8000/docs
```

## Quick Start (Google Colab)

Open each notebook in order (`01` → `05`). Notebook 5 runs the unified FastAPI
service and exposes a public URL via **pyngrok**.

```python
!pip install pyngrok fastapi uvicorn nest_asyncio -q
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)
```

---

## API Endpoints (Day 5)

| Method | Path                  | Purpose |
|--------|-----------------------|---------|
| POST   | `/classify-product`   | Upload product image → category |
| POST   | `/analyze-sentiment`  | Review text → Positive / Neutral / Negative |
| POST   | `/chatbot`            | Customer query → automated reply |
| POST   | `/identify-customer`  | Face snapshot → recognised customer ID |
| GET    | `/health`             | Liveness probe |

Interactive docs at `/docs` (Swagger UI) once the server is running.

---

## Datasets

Small **sample** datasets are bundled under `data/` so every notebook executes
end-to-end without external downloads. For production accuracy, swap in:

* **Products** → [Fashion Product Images (Kaggle)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
* **Reviews**  → [Amazon Customer Reviews](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)
* **Faces**    → [LFW – Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)

---

## License

MIT — free for educational and commercial use.
