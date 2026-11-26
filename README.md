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
├── artifacts/                  # Auto-created during training (models, confusion matrices, metrics.json)
├── requirements.txt
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
