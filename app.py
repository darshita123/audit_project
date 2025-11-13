# app.py
import streamlit as st
import pandas as pd
import os, json
from crew_setup import run_audit_query
from database.mongo_client import get_database
from agents.fine_tuner_agent import fine_tune_local_model
from agents.reviewer_agent import simple_review_check

st.set_page_config(page_title="Audit AI", layout="wide")
st.title("Local Audit Intelligence System")

db = get_database()
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload bank CSV", type=["csv"])

# -----------------------------
# Run Analysis and Q&A
# -----------------------------
if st.sidebar.button("Run Analysis and Q&A"):
    if not uploaded_file:
        st.sidebar.error("Upload a CSV first.")
    else:
        path = os.path.join("datasets", uploaded_file.name)
        os.makedirs("datasets", exist_ok=True)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        result = run_audit_query(path)
        st.success("✅ Run complete!")

        st.subheader("📄 Labeled Data Preview")
        df = pd.read_csv(result["labeled_file"])
        st.dataframe(df.head())

        if result.get("qa_file") and os.path.exists(result["qa_file"]):
            st.subheader("💬 Generated Q&A")
            try:
                with open(result["qa_file"], "r") as f:
                    qa_data = json.load(f)
                st.json(qa_data[-5:])
            except Exception as e:
                st.error(f"Could not load Q&A JSON: {e}")

        # Run reviewer
        st.subheader("🔍 Review Summary")
        review = simple_review_check(result["labeled_file"])
        st.json(review)

        # ✅ Conditional fine-tuning button
        if review and review.get("accuracy", 1) < 0.85:
            st.warning("⚠ Review score is below threshold. Fine-tuning recommended.")
            if st.button("Run Fine-tuning for This Dataset"):
                with st.spinner("Fine-tuning model for current dataset..."):
                    out = fine_tune_local_model(mode="auto")
                st.success(f"Fine-tuning complete. Model saved to {out}")
        else:
            st.info("✅ Model performance is good. No fine-tuning needed.")

# -----------------------------
# ✅ Global fine-tuning (manual)
# -----------------------------
if st.sidebar.button("Run Global Fine-tuning"):
    st.sidebar.info("Starting global fine-tuning on all accumulated Q&A data...")
    with st.spinner("Fine-tuning model on full dataset..."):
        out = fine_tune_local_model(mode="global")
    st.sidebar.success(f"Global fine-tuning finished. Model saved to {out}")

# -----------------------------
# Logs
# -----------------------------
if st.sidebar.button("Show Recent Logs"):
    logs = list(db.logs.find().sort("timestamp", -1).limit(10))
    st.subheader("🧾 Recent Logs")
    for l in logs:
        st.write(f"[{l['timestamp']}] {l['agent']} → {l['action']}")
        st.json(l["details"])