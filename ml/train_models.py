import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

os.makedirs("artifacts", exist_ok=True)
os.makedirs("artifacts/distilbert_model", exist_ok=True)

LABELS = ["Sizing", "Damaged Item", "Wrong Product", "Late Delivery"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    df = pd.read_csv("data/return_reasons.csv")
    df["clean_comment"] = df["comment"].apply(clean_text)
    df["label"] = df["return_reason"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 1. Baseline: TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train = tfidf.fit_transform(train_df["clean_comment"])
    X_test = tfidf.transform(test_df["clean_comment"])

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, train_df["label"])
    lr_preds = lr_model.predict(X_test)

    lr_metrics = {
        "accuracy": round(float(accuracy_score(test_df["label"], lr_preds)), 4),
        "macro_f1": round(float(f1_score(test_df["label"], lr_preds, average="macro")), 4),
        "confusion_matrix": confusion_matrix(test_df["label"], lr_preds).tolist(),
        "report": classification_report(test_df["label"], lr_preds, target_names=LABELS, output_dict=True),
    }

    joblib.dump(tfidf, "artifacts/tfidf_vectorizer.joblib")
    joblib.dump(lr_model, "artifacts/logreg_model.joblib")

    # 2. Transformer: Fine-tuned DistilBERT
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def tokenize_fn(batch):
        return tokenizer(batch["clean_comment"], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["clean_comment", "label"]], preserve_index=False).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df[["clean_comment", "label"]], preserve_index=False).map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="./artifacts/checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()

    pred_output = trainer.predict(test_ds)
    bert_preds = np.argmax(pred_output.predictions, axis=1)

    bert_metrics = {
        "accuracy": round(float(accuracy_score(test_df["label"], bert_preds)), 4),
        "macro_f1": round(float(f1_score(test_df["label"], bert_preds, average="macro")), 4),
        "confusion_matrix": confusion_matrix(test_df["label"], bert_preds).tolist(),
        "report": classification_report(test_df["label"], bert_preds, target_names=LABELS, output_dict=True),
    }

    bert_model.save_pretrained("artifacts/distilbert_model")
    tokenizer.save_pretrained("artifacts/distilbert_model")

    summary = {
        "labels": LABELS,
        "tfidf_logreg": lr_metrics,
        "distilbert": bert_metrics,
    }
    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete! Metrics saved to artifacts/metrics.json")
    print(f"TF-IDF + LR Accuracy: {lr_metrics['accuracy']} | Macro F1: {lr_metrics['macro_f1']}")
    print(f"DistilBERT Accuracy:  {bert_metrics['accuracy']} | Macro F1: {bert_metrics['macro_f1']}")

if __name__ == "__main__":
    main()
