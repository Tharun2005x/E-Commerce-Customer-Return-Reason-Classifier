import io
import os
import sys
import json
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(
    page_title="ReturnSense AI | E-Commerce Return Reason Classifier",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Modern UI Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #3B82F6 100%);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(6px);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        color: #E0F2FE;
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin: 0;
    }

    /* Styled KPI Cards */
    .kpi-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-left: 4px solid #3B82F6;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94A3B8;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #F8FAFC;
    }

    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(59, 130, 246, 0.12) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }
    .reason-pill {
        display: inline-block;
        padding: 0.35rem 0.95rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        color: #FFFFFF;
        margin-right: 0.75rem;
    }

    /* Sidebar Category Pills */
    .cat-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
    }

    /* Tabs Polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.45);
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Category Color & Icon Mapping ---
CATEGORY_META = {
    "Sizing": {"icon": "📏", "color": "#3B82F6", "action": "Flag apparel size chart & fit guide"},
    "Damaged Item": {"icon": "💔", "color": "#EF4444", "action": "Escalate to Warehouse QA & packaging team"},
    "Wrong Product": {"icon": "📦", "color": "#F59E0B", "action": "Audit SKU barcode & fulfillment picking"},
    "Late Delivery": {"icon": "🚚", "color": "#8B5CF6", "action": "Trigger logistics SLA delay alert"},
}

# --- Resolve Backend Mode (Remote FastAPI OR Standalone Cloud Engine) ---
try:
    SECRET_API_URL = st.secrets.get("API_URL", "")
except Exception:
    SECRET_API_URL = ""

API_URL = (SECRET_API_URL or os.getenv("API_URL", "")).strip().rstrip("/")

USE_REMOTE_API = False
if API_URL:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            USE_REMOTE_API = True
    except Exception:
        USE_REMOTE_API = False
else:
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


# --- Unified Data Access Layer ---
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


# --- Sidebar Control Panel ---
with st.sidebar:
    st.markdown("## 🎛️ Control Center")
    model_choice = st.selectbox(
        "Active NLP Engine",
        options=["tfidf_logreg", "distilbert"],
        format_func=lambda x: (
            "⚡ TF-IDF + Logistic Regression"
            if x == "tfidf_logreg"
            else "🧠 DistilBERT Transformer"
        ),
    )

    st.markdown("---")
    st.markdown("### 🏷️ Root-Cause Buckets")
    for label, meta in CATEGORY_META.items():
        st.markdown(
            f"""
            <div class="cat-pill" style="border-left: 4px solid {meta['color']};">
                <span>{meta['icon']} {label}</span>
                <span style="color:{meta['color']}; font-size:0.75rem;">● ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    status_text = f"FastAPI ({API_URL})" if USE_REMOTE_API else "Standalone Cloud Engine"
    st.caption(f"🟢 **Runtime:** `{status_text}`")
    st.caption("🗄️ **Storage:** `SQLite (returns.db)`")


# --- Hero Banner ---
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">NLP-10 • Automated E-Commerce Operations Intelligence</div>
        <div class="hero-title">📦 ReturnSense AI — Return Reason Classifier</div>
        <div class="hero-subtitle">
            Real-time customer return triage powered by <b>TF-IDF + Logistic Regression</b> &amp; <b>DistilBERT</b> with automated SQLite audit logging.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Main Navigation Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Return Triage",
    "📂 Batch CSV Upload",
    "🗄️ Operations Database",
    "📊 Model Benchmark & Matrix",
])

# --- TAB 1: SINGLE PREDICTION ---
with tab1:
    st.markdown("#### Quick-Fill Demo Presets")
    p1, p2, p3, p4 = st.columns(4)
    if "preset_oid" not in st.session_state:
        st.session_state.preset_oid = "ORD-702"
        st.session_state.preset_txt = "I ordered a blue jacket in medium, but they sent me a red sweater instead."

    if p1.button("📏 Test Sizing", use_container_width=True):
        st.session_state.preset_oid = "ORD-701"
        st.session_state.preset_txt = "The dress runs way too tight around the shoulders and waist."
    if p2.button("💔 Test Damaged", use_container_width=True):
        st.session_state.preset_oid = "ORD-703"
        st.session_state.preset_txt = "Zipper broke on first use and the side seam arrived torn."
    if p3.button("📦 Test Wrong Item", use_container_width=True):
        st.session_state.preset_oid = "ORD-702"
        st.session_state.preset_txt = "I ordered a blue jacket in medium, but they sent me a red sweater instead."
    if p4.button("🚚 Test Late Delivery", use_container_width=True):
        st.session_state.preset_oid = "ORD-704"
        st.session_state.preset_txt = "Package arrived 8 days after the promised delivery date."

    st.markdown("---")
    in_col1, in_col2 = st.columns([1, 2.4])
    with in_col1:
        order_id = st.text_input("Order ID", value=st.session_state.preset_oid)
    with in_col2:
        comment = st.text_area("Customer Return Comment", value=st.session_state.preset_txt, height=95)

    if st.button("🚀 Classify & Log Return", type="primary"):
        if not comment.strip():
            st.warning("Please enter a customer return comment.")
        else:
            try:
                with st.spinner("Analyzing customer return intent..."):
                    data = api_predict_single(order_id, comment, model_choice)

                reason = data["predicted_reason"]
                conf_pct = data["confidence"] * 100
                meta = CATEGORY_META.get(reason, {"icon": "🏷️", "color": "#3B82F6", "action": "Route to support"})

                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="reason-pill" style="background:{meta['color']};">
                            {meta['icon']} {reason}
                        </span>
                        <span style="font-size:1.05rem; font-weight:700;">
                            Confidence: {conf_pct:.2f}%
                        </span>
                        <div style="margin-top:0.65rem; font-size:0.9rem; color:#CBD5E1;">
                            <b>Recommended Ops Action:</b> {meta['action']} &nbsp;|&nbsp; <b>Logged Order:</b> <code>{data['order_id']}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                prob_df = pd.DataFrame({
                    "Return Reason": list(data["probabilities"].keys()),
                    "Probability": list(data["probabilities"].values()),
                }).sort_values("Probability", ascending=True)

                fig, ax = plt.subplots(figsize=(7.5, 2.7))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                bar_colors = [CATEGORY_META.get(r, {}).get("color", "#3B82F6") for r in prob_df["Return Reason"]]
                bars = ax.barh(prob_df["Return Reason"], prob_df["Probability"], color=bar_colors, height=0.58)
                ax.set_xlim(0, 1.12)
                ax.tick_params(colors="#E2E8F0", labelsize=9.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#475569")
                ax.spines["bottom"].set_color("#475569")
                ax.set_title("Class Confidence Distribution", color="#F8FAFC", fontsize=10.5, fontweight="bold", loc="left")
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w*100:.1f}%", va="center", color="#F8FAFC", fontweight="bold", fontsize=9)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Prediction error: {e}")

# --- TAB 2: BATCH CSV UPLOAD ---
with tab2:
    st.markdown("#### Bulk Return Spreadsheet Triage")
    st.info("Upload a `.csv` file with a `comment` (or `review`) column and optional `order_id` column to classify hundreds of returns at once.")

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

    uploaded = st.file_uploader("Drop Return Comments CSV Here", type=["csv"])
    if uploaded and st.button("⚡ Run Batch Classification", type="primary"):
        try:
            with st.spinner("Processing batch rows & committing to SQLite..."):
                out = api_predict_batch(uploaded, model_choice)

            st.success(f"Successfully classified and logged **{out['total_processed']}** return records!")
            res_df = pd.DataFrame(out["results"])
            st.dataframe(res_df, use_container_width=True)

            st.download_button(
                "⬇️ Download Labeled Results CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="categorized_returns.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Batch processing error: {e}")

# --- TAB 3: SQLITE DATABASE VIEW ---
with tab3:
    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.markdown("#### Live Operations Audit Log (`returns.db`)")
    with h_col2:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            try:
                api_clear_records()
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear database: {e}")

    try:
        records = api_get_records(limit=200)
        if not records:
            st.info("No return records logged yet. Run a prediction in Tab 1 or Tab 2 to populate SQLite.")
        else:
            df_rec = pd.DataFrame(records)
            top_driver = df_rec["predicted_reason"].value_counts().idxmax()
            avg_conf = df_rec["confidence"].mean() * 100

            k1, k2, k3 = st.columns(3)
            k1.markdown(
                f'<div class="kpi-card"><div class="kpi-label">Total Processed Returns</div><div class="kpi-value">{len(df_rec)}</div></div>',
                unsafe_allow_html=True,
            )
            k2.markdown(
                f'<div class="kpi-card" style="border-left-color:#10B981;"><div class="kpi-label">Top Return Driver</div><div class="kpi-value">{top_driver}</div></div>',
                unsafe_allow_html=True,
            )
            k3.markdown(
                f'<div class="kpi-card" style="border-left-color:#8B5CF6;"><div class="kpi-label">Avg Model Confidence</div><div class="kpi-value">{avg_conf:.1f}%</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.bar_chart(df_rec["predicted_reason"].value_counts(), color="#3B82F6")
            st.dataframe(df_rec, use_container_width=True)
    except Exception as e:
        st.error(f"Database error: {e}")

# --- TAB 4: MODEL COMPARISON & CONFUSION MATRICES ---
with tab4:
    st.markdown("#### Architecture Benchmark: Classical ML vs. Transformer")
    try:
        metrics = api_get_metrics()
        labels = metrics["labels"]

        m_col1, m_col2 = st.columns(2)
        m_col1.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:#3B82F6;">
                <div class="kpi-label">TF-IDF + Logistic Regression</div>
                <div class="kpi-value">{metrics['tfidf_logreg']['accuracy']*100:.2f}% Acc &nbsp;|&nbsp; {metrics['tfidf_logreg']['macro_f1']*100:.2f}% F1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m_col2.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:#8B5CF6;">
                <div class="kpi-label">DistilBERT Fine-Tuned Transformer</div>
                <div class="kpi-value">{metrics['distilbert']['accuracy']*100:.2f}% Acc &nbsp;|&nbsp; {metrics['distilbert']['macro_f1']*100:.2f}% F1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
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
            st.markdown("**DistilBERT Transformer — Confusion Matrix**")
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