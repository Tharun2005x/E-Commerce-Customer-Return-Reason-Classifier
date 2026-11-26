import io
import os
import sys
import json
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is importable for standalone Streamlit Cloud mode
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(
    page_title="NLP-10 | E-Commerce Return Reason Classifier",
    page_icon="📦",
    layout="wide",
)

# Resolve API_URL from Streamlit Secrets or Environment Variables (optional)
try:
    SECRET_API_URL = st.secrets.get("API_URL", "")
except Exception:
    SECRET_API_URL = ""

API_URL = (SECRET_API_URL or os.getenv("API_URL", "")).strip().rstrip("/")

# Determine whether to use remote FastAPI or direct in-process backend
USE_REMOTE_API = False
if API_URL:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            USE_REMOTE_API = True
    except Exception:
        USE_REMOTE_API = False
else:
    # Check if local FastAPI is running on port 8000
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=1)
        if r.status_code == 200:
            API_URL = "http://127.0.0.1:8000"
            USE_REMOTE_API = True
    except Exception:
        USE_REMOTE_API = False

if not USE_REMOTE_API:
    from backend.database import init_db, SessionLocal, ReturnRecord
    from backend.main import run_inference
    init_db()


# --- Unified Data Access Layer (Remote FastAPI OR Direct Python Backend) ---

def api_predict_single(order_id: str, comment: str, model_type: str) -> dict:
    if USE_REMOTE_API:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"order_id": order_id, "comment": comment, "model_type": model_type},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    reason, conf, prob_map = run_inference(comment, model_type)
    db = SessionLocal()
    try:
        record = ReturnRecord(
            order_id=order_id or "ORD-SINGLE",
            comment=comment,
            predicted_reason=reason,
            confidence=conf,
            model_used=model_type,
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
            "model_used": model_type,
        }
    finally:
        db.close()


def api_predict_batch(uploaded_file, model_type: str) -> dict:
    if USE_REMOTE_API:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        resp = requests.post(
            f"{API_URL}/predict-batch",
            files=files,
            data={"model_type": model_type},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    text_col = next(
        (c for c in df.columns if c.lower() in ["comment", "review", "review text", "reason", "text"]),
        None,
    )
    if not text_col:
        raise ValueError("CSV must contain a 'comment' or 'review' column.")

    order_col = next((c for c in df.columns if c.lower() in ["order_id", "orderid", "id"]), None)

    db = SessionLocal()
    results = []
    try:
        for idx, row in df.iterrows():
            comment_val = str(row[text_col]).strip()
            if not comment_val or comment_val.lower() == "nan":
                continue
            oid = str(row[order_col]) if order_col else f"BATCH-{idx+1}"
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
    finally:
        db.close()

    return {"total_processed": len(results), "results": results}


def api_get_records(limit: int = 200) -> list:
    if USE_REMOTE_API:
        resp = requests.get(f"{API_URL}/records", params={"limit": limit}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    db = SessionLocal()
    try:
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
    finally:
        db.close()


def api_clear_records():
    if USE_REMOTE_API:
        resp = requests.delete(f"{API_URL}/records", timeout=15)
        resp.raise_for_status()
        return resp.json()

    db = SessionLocal()
    try:
        db.query(ReturnRecord).delete()
        db.commit()
    finally:
        db.close()


def api_get_metrics() -> dict:
    if USE_REMOTE_API:
        resp = requests.get(f"{API_URL}/metrics", timeout=15)
        resp.raise_for_status()
        return resp.json()

    metrics_path = os.path.join(ROOT_DIR, "artifacts", "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError("artifacts/metrics.json not found.")
    with open(metrics_path) as f:
        return json.load(f)


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Active Inference Model",
    options=["distilbert", "tfidf_logreg"],
    format_func=lambda x: "DistilBERT (Transformer)" if x == "distilbert" else "TF-IDF + Logistic Regression",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Target Root Causes:**")
st.sidebar.markdown(
    "- 📏 `Sizing`\n"
    "- 💔 `Damaged Item`\n"
    "- 📦 `Wrong Product`\n"
    "- 🚚 `Late Delivery`"
)
st.sidebar.markdown("---")
mode_label = f"FastAPI ({API_URL})" if USE_REMOTE_API else "Standalone Cloud Engine (SQLite)"
st.sidebar.caption(f"Backend Mode: `{mode_label}`")

# --- Header ---
st.title("📦 E-Commerce Customer Return Reason Classifier")
st.caption("NLP-10 • Automated Root-Cause Triage (TF-IDF + Logistic Regression vs. DistilBERT) • FastAPI + SQLite")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Return Triage",
    "📂 Batch CSV Upload",
    "🗄️ Operations Database",
    "📊 Model Comparison & Confusion Matrix",
])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.subheader("Classify Single Customer Return Comment")
    col1, col2 = st.columns([1, 2])
    with col1:
        order_id = st.text_input("Order ID", value="ORD-702")
    with col2:
        comment = st.text_area(
            "Customer Return Comment",
            value="I ordered a blue jacket in medium, but they sent me a red sweater instead.",
            height=100,
        )

    if st.button("Classify & Save to DB", type="primary"):
        if not comment.strip():
            st.warning("Please enter a customer return comment.")
        else:
            try:
                with st.spinner("Running NLP inference..."):
                    data = api_predict_single(order_id, comment, model_choice)

                st.success(f"**Predicted Root Cause:** `{data['predicted_reason']}`  |  **Confidence:** `{data['confidence']*100:.2f}%`")

                prob_df = pd.DataFrame({
                    "Return Reason": list(data["probabilities"].keys()),
                    "Probability": list(data["probabilities"].values()),
                }).sort_values("Probability", ascending=True)

                fig, ax = plt.subplots(figsize=(7, 2.8))
                bars = ax.barh(prob_df["Return Reason"], prob_df["Probability"], color="#2563EB")
                ax.set_xlim(0, 1.05)
                ax.set_xlabel("Confidence Probability")
                ax.set_title("Class Probability Distribution")
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%", va="center", fontsize=9)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# --- TAB 2: BATCH CSV UPLOAD ---
with tab2:
    st.subheader("Batch CSV Return Processing")
    st.info("Upload a CSV containing a `comment` (or `review`) column and optional `order_id` column.")

    sample_df = pd.DataFrame({
        "order_id": ["ORD-801", "ORD-802", "ORD-803", "ORD-804"],
        "comment": [
            "The dress runs way too small around the waist.",
            "Glass jar arrived shattered inside the box.",
            "I ordered black sneakers but got white sandals.",
            "Delivery took 12 days instead of 2 days.",
        ],
    })
    st.download_button(
        "⬇️ Download Sample Returns CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_returns.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Return Comments CSV", type=["csv"])
    if uploaded and st.button("Run Batch Classification", type="primary"):
        try:
            with st.spinner("Classifying batch comments..."):
                out = api_predict_batch(uploaded, model_choice)

            st.success(f"Processed and logged **{out['total_processed']}** return records!")
            res_df = pd.DataFrame(out["results"])
            st.dataframe(res_df, use_container_width=True)

            st.download_button(
                "⬇️ Download Categorized Results CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="categorized_returns.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch processing error: {e}")

# --- TAB 3: SQLITE DATABASE VIEW ---
with tab3:
    st.subheader("Stored Return Classifications (SQLite)")
    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("🗑️ Clear DB Logs"):
            try:
                api_clear_records()
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear database: {e}")

    try:
        records = api_get_records(limit=200)
        if not records:
            st.info("No return records logged yet. Classify a single comment or upload a CSV to populate SQLite.")
        else:
            df_rec = pd.DataFrame(records)
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Logged Returns", len(df_rec))
            k2.metric("Top Return Driver", df_rec["predicted_reason"].value_counts().idxmax())
            k3.metric("Avg Model Confidence", f"{df_rec['confidence'].mean()*100:.1f}%")

            st.bar_chart(df_rec["predicted_reason"].value_counts())
            st.dataframe(df_rec, use_container_width=True)
    except Exception as e:
        st.error(f"Database error: {e}")

# --- TAB 4: MODEL COMPARISON & CONFUSION MATRICES ---
with tab4:
    st.subheader("Benchmark Comparison: TF-IDF + Logistic Regression vs. DistilBERT")
    try:
        metrics = api_get_metrics()
        labels = metrics["labels"]

        comp_df = pd.DataFrame([
            {
                "Model": "TF-IDF + Logistic Regression (Baseline)",
                "Test Accuracy": f"{metrics['tfidf_logreg']['accuracy']*100:.2f}%",
                "Macro F1-Score": f"{metrics['tfidf_logreg']['macro_f1']*100:.2f}%",
            },
            {
                "Model": "DistilBERT (Fine-Tuned Transformer)",
                "Test Accuracy": f"{metrics['distilbert']['accuracy']*100:.2f}%",
                "Macro F1-Score": f"{metrics['distilbert']['macro_f1']*100:.2f}%",
            },
        ])
        st.table(comp_df)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**TF-IDF + Logistic Regression — Confusion Matrix**")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                metrics["tfidf_logreg"]["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1,
            )
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")
            st.pyplot(fig1)
            plt.close(fig1)

        with col_b:
            st.markdown("**DistilBERT — Confusion Matrix**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                metrics["distilbert"]["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Purples",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2,
            )
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)
            plt.close(fig2)
    except Exception as e:
        st.error(f"Could not load evaluation metrics: {e}")