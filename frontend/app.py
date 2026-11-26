import os
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

raw_api_url = os.getenv("API_URL", "http://127.0.0.1:8000").strip().rstrip("/")
if raw_api_url and not raw_api_url.startswith(("http://", "https://")):
    raw_api_url = f"http://{raw_api_url}"
API_URL = raw_api_url


def call_api(method: str, endpoint: str, **kwargs):
    """Send an HTTP request to the FastAPI backend and return (response, error_message)."""
    timeout = kwargs.pop("timeout", 30)
    try:
        res = requests.request(method, f"{API_URL}{endpoint}", timeout=timeout, **kwargs)
        return res, None
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach backend API at `{API_URL}` ({exc}). Start it with `uvicorn backend.main:app --reload --port 8000`."


def extract_error_detail(res, fallback: str) -> str:
    """Safely parse error detail from an API response."""
    try:
        return res.json().get("detail", fallback)
    except ValueError:
        return res.text or fallback


def get_available_models():
    """Fetch models available on the backend API."""
    res, err = call_api("GET", "/health", timeout=2)
    if not err and res.status_code == 200:
        data = res.json()
        models = data.get("available_models")
        if isinstance(models, list) and models:
            return models
    return ["distilbert", "tfidf_logreg"]


st.set_page_config(
    page_title="E-Commerce Return Reason Classifier",
    page_icon="📦",
    layout="wide",
)

available_models = get_available_models()

st.title("📦 E-Commerce Customer Return Reason Classifier")
st.caption("NLP-10 • Automated Root-Cause Triage (TF-IDF + Logistic Regression vs. DistilBERT) • FastAPI + SQLite")

with st.sidebar:
    st.header("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Active Inference Model",
        options=available_models,
        format_func=lambda x: "DistilBERT (Transformer)" if x == "distilbert" else "TF-IDF + Logistic Regression",
    )
    if "distilbert" not in available_models:
        st.info("Running lightweight `TF-IDF + Logistic Regression` mode (DistilBERT weights are omitted in cloud deployment).")
    st.markdown("---")
    st.markdown("**Target Root Causes:**")
    st.markdown("- 📏 `Sizing`\n- 💔 `Damaged Item`\n- 📦 `Wrong Product`\n- 🚚 `Late Delivery`")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Return Triage",
    "📂 Batch CSV Upload",
    "🗄️ Operations Database",
    "📊 Model Comparison & Confusion Matrix",
])

with tab1:
    st.subheader("Classify Single Customer Return Comment")
    col1, col2 = st.columns([1, 3])
    with col1:
        order_id = st.text_input("Order ID", value="ORD-2026-884")
    with col2:
        comment = st.text_area(
            "Customer Return Comment",
            placeholder="e.g., Ordered size M but the sleeves are super tight and the zipper arrived broken...",
            height=100,
        )

    if st.button("Classify & Save to DB", type="primary"):
        if not comment.strip():
            st.warning("Please enter a return comment.")
        else:
            res, err = call_api(
                "POST",
                "/predict",
                json={"order_id": order_id, "comment": comment, "model_type": model_choice},
            )
            if err:
                st.error(err)
            elif res.status_code == 200:
                data = res.json()
                st.success(f"**Root Cause:** `{data['predicted_reason']}` (Confidence: **{data['confidence']*100:.2f}%**)")

                prob_df = pd.DataFrame(
                    list(data["probabilities"].items()), columns=["Root Cause", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Root Cause"))
            else:
                st.error(extract_error_detail(res, "API Error"))

with tab2:
    st.subheader("Batch CSV Return Processing")
    st.info("Upload a CSV containing a `comment` (or `review`) column and optional `order_id` column.")

    sample_csv = pd.DataFrame({
        "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104"],
        "comment": [
            "The dress was way too tight around the waist and hips.",
            "Received a green sweater instead of the black jacket I ordered.",
            "Package arrived 10 days after the guaranteed delivery date.",
            "The seam on the left sleeve was completely torn open.",
        ],
    })
    st.download_button(
        "⬇️ Download Sample Returns CSV",
        data=sample_csv.to_csv(index=False).encode("utf-8"),
        file_name="sample_returns.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Return Comments CSV", type=["csv"])
    if uploaded_file and st.button("Run Batch Classification", type="primary"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        res, err = call_api("POST", "/predict-batch", files=files, data={"model_type": model_choice})
        if err:
            st.error(err)
        elif res.status_code == 200:
            payload = res.json()
            st.success(f"Processed and stored **{payload['total_processed']}** return records in SQLite.")
            out_df = pd.DataFrame(payload["results"])
            st.dataframe(out_df, width="stretch")
            st.download_button(
                "⬇️ Download Categorized Results CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="categorized_returns.csv",
                mime="text/csv",
            )
        else:
            st.error(extract_error_detail(res, "Batch processing failed."))

with tab3:
    st.subheader("Stored Return Classifications (SQLite)")
    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("🗑️ Clear DB Logs"):
            del_res, del_err = call_api("DELETE", "/records")
            if del_err:
                st.error(del_err)
            elif del_res.status_code == 200:
                st.rerun()
            else:
                st.error(extract_error_detail(del_res, "Failed to clear records."))

    rec_res, rec_err = call_api("GET", "/records")
    if rec_err:
        st.error(rec_err)
    elif rec_res.status_code == 200 and rec_res.json():
        records_df = pd.DataFrame(rec_res.json())
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Logged Returns", len(records_df))
        m2.metric("Top Return Driver", records_df["predicted_reason"].mode()[0])
        m3.metric("Avg Confidence", f"{records_df['confidence'].mean()*100:.1f}%")

        st.bar_chart(records_df["predicted_reason"].value_counts())
        st.dataframe(records_df, width="stretch")
    elif rec_res.status_code == 200:
        st.info("No records logged in SQLite yet. Run a single or batch prediction first.")
    else:
        st.error(extract_error_detail(rec_res, "Failed to fetch records."))

with tab4:
    st.subheader("TF-IDF + Logistic Regression vs. DistilBERT Evaluation")
    m_res, m_err = call_api("GET", "/metrics")
    if m_err:
        st.error(m_err)
    elif m_res.status_code == 200:
        metrics = m_res.json()
        labels = metrics["labels"]
        lr = metrics["tfidf_logreg"]
        bert = metrics["distilbert"]

        comp_df = pd.DataFrame({
            "Model": ["TF-IDF + Logistic Regression", "DistilBERT (Fine-Tuned)"],
            "Test Accuracy": [lr["accuracy"], bert["accuracy"]],
            "Macro F1-Score": [lr["macro_f1"], bert["macro_f1"]],
        })
        st.table(comp_df)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**TF-IDF + Logistic Regression Confusion Matrix**")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            sns.heatmap(lr["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1)
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")
            st.pyplot(fig1)
            plt.close(fig1)

        with col_b:
            st.markdown("**DistilBERT Confusion Matrix**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            sns.heatmap(bert["confusion_matrix"], annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=ax2)
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.warning("Metrics not found. Run `python ml/train_models.py` first.")
