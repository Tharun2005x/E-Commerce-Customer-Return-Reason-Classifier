import io
import os
import re
import json
import joblib
import torch
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.database import init_db, get_db, ReturnRecord

app = FastAPI(title="E-Commerce Customer Return Reason Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = ["Sizing", "Damaged Item", "Wrong Product", "Late Delivery"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
VALID_MODELS = {"distilbert", "tfidf_logreg"}

try:
    tfidf = joblib.load("artifacts/tfidf_vectorizer.joblib")
    lr_model = joblib.load("artifacts/logreg_model.joblib")
    tokenizer = AutoTokenizer.from_pretrained("artifacts/distilbert_model")
    bert_model = AutoModelForSequenceClassification.from_pretrained("artifacts/distilbert_model")
    bert_model.eval()
except Exception as exc:
    raise RuntimeError(
        "Model artifacts not found or invalid in 'artifacts/'. Run 'python ml/train_models.py' first."
    ) from exc

init_db()

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_inference(text: str, model_type: str):
    if model_type not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model_type '{model_type}'. Choose from {sorted(VALID_MODELS)}.",
        )
    cleaned = clean_text(text)
    if model_type == "tfidf_logreg":
        vec = tfidf.transform([cleaned])
        probs = lr_model.predict_proba(vec)[0]
        pred_idx = int(probs.argmax())
        prob_map = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
        return LABELS[pred_idx], round(float(probs[pred_idx]), 4), prob_map
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred_idx = int(probs.argmax())
    prob_map = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
    return LABELS[pred_idx], round(float(probs[pred_idx]), 4), prob_map

class PredictRequest(BaseModel):
    order_id: Optional[str] = "ORD-1001"
    comment: str
    model_type: str = "distilbert"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    if not os.path.exists("artifacts/metrics.json"):
        raise HTTPException(status_code=404, detail="Metrics file not found. Run training first.")
    with open("artifacts/metrics.json", encoding="utf-8") as f:
        return json.load(f)

@app.post("/predict")
def predict_single(req: PredictRequest, db: Session = Depends(get_db)):
    if not req.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")

    reason, conf, prob_map = run_inference(req.comment, req.model_type)

    record = ReturnRecord(
        order_id=req.order_id or "ORD-SINGLE",
        comment=req.comment,
        predicted_reason=reason,
        confidence=conf,
        model_used=req.model_type,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "order_id": record.order_id,
        "comment": record.comment,
        "predicted_reason": reason,
        "confidence": conf,
        "probabilities": prob_map,
        "model_used": req.model_type,
    }

@app.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_type: str = Form("distilbert"),
    db: Session = Depends(get_db),
):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file.")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    text_col = next((c for c in df.columns if c.lower() in ["comment", "review", "review text", "reason", "text"]), None)
    if not text_col:
        raise HTTPException(status_code=400, detail="CSV must contain a 'comment' or 'review' column.")

    order_col = next((c for c in df.columns if c.lower() in ["order_id", "orderid", "id"]), None)

    results = []
    for idx, row in df.iterrows():
        raw_comment = row[text_col]
        if pd.isna(raw_comment):
            continue
        comment_val = str(raw_comment).strip()
        if not comment_val:
            continue
        raw_oid = row[order_col] if order_col else None
        oid = str(raw_oid).strip() if (order_col and not pd.isna(raw_oid) and str(raw_oid).strip()) else f"BATCH-{idx+1}"
        reason, conf, _ = run_inference(comment_val, model_type)

        rec = ReturnRecord(
            order_id=oid,
            comment=comment_val,
            predicted_reason=reason,
            confidence=conf,
            model_used=model_type,
        )
        db.add(rec)
        results.append({
            "order_id": oid,
            "comment": comment_val,
            "predicted_reason": reason,
            "confidence": conf,
            "model_used": model_type,
        })

    db.commit()
    return {"total_processed": len(results), "results": results}

@app.get("/records")
def list_records(limit: int = 200, db: Session = Depends(get_db)):
    records = db.query(ReturnRecord).order_by(ReturnRecord.id.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "order_id": r.order_id,
            "comment": r.comment,
            "predicted_reason": r.predicted_reason,
            "confidence": r.confidence,
            "model_used": r.model_used,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]

@app.delete("/records")
def clear_records(db: Session = Depends(get_db)):
    db.query(ReturnRecord).delete()
    db.commit()
    return {"message": "All return records cleared."}
