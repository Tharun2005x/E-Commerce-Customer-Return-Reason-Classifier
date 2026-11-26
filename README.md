# NLP-10: E-Commerce Customer Return Reason Classifier

An end-to-end NLP multi-class classification system that categorizes free-text customer return comments into structured operational root causes (**Sizing**, **Damaged Item**, **Wrong Product**, **Late Delivery**) with model comparison (**TF-IDF + Logistic Regression** vs. **Fine-tuned DistilBERT**), a **FastAPI** REST backend, **SQLite** database persistence, and a **Streamlit** operations dashboard.

## Project Structure

```text
Ecommerce-Return-Reason-Classifier/
├── data/
│   └── generate_data.py        # Builds clean labeled dataset (Women's Clothing Reviews + synthetic returns)
├── ml/
│   └── train_models.py         # Trains TF-IDF + Logistic Regression & DistilBERT, exports metrics & models
├── backend/
│   ├── __init__.py
│   ├── database.py             # SQLite + SQLAlchemy ORM models & session manager
│   └── main.py                 # FastAPI REST API (single predict, batch CSV upload, DB logs, metrics)
├── frontend/
│   └── app.py                  # Streamlit Operations Dashboard (Live Triage, Batch CSV, DB Logs, Eval)
├── artifacts/                  # Pre-trained TF-IDF + LR & metrics.json (DistilBERT generated via train_models.py)
├── render.yaml                 # Render Blueprint configuration
├── requirements.txt            # Full dependencies (local training + DistilBERT + serving)
├── requirements-render.txt     # Lightweight runtime dependencies for Render Free Tier
├── start.py                    # Combined FastAPI + Streamlit launcher for Render Web Service
└── README.md
```

## Setup & Execution (Windows PowerShell)

1. **Create & activate virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**
   ```powershell
   python data/generate_data.py
   ```

3. **Train baseline & DistilBERT models:**
   ```powershell
   python ml/train_models.py
   ```

4. **Start the FastAPI backend + SQLite database (Terminal 1):**
   ```powershell
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Launch the Streamlit frontend (Terminal 2):**
   ```powershell
   streamlit run frontend/app.py
   ```

## Deploying to Render

This repository includes `render.yaml`, `requirements-render.txt`, and `start.py` configured for Render's Free Tier (512 MB RAM):

- **Option 1 (Blueprint):** In Render, select **New + → Blueprint**, connect this GitHub repo, and deploy (`render.yaml` configures everything automatically).
- **Option 2 (Manual Web Service):**
  - **Runtime:** Python 3
  - **Build Command:** `pip install -r requirements-render.txt`
  - **Start Command:** `python start.py`
  - **Environment Variables:** `ENABLE_DISTILBERT=false`

> **Note:** `artifacts/distilbert_model/` (255 MB) is excluded from Git due to GitHub's 100 MB limit and Render's 512 MB RAM limit; the deployed app automatically uses the bundled `TF-IDF + Logistic Regression` model while displaying both models' evaluation metrics in the comparison tab.
