import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="SpamShield AI SaaS",
    page_icon="📧",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "user" not in st.session_state:
    st.session_state.user = "Guest User"

# =========================
# SIDEBAR (SAAS DASHBOARD)
# =========================
with st.sidebar:
    st.title("📧 SpamShield AI")
    st.markdown("### SaaS Dashboard")

    st.markdown("---")

    st.text_input("👤 User Name", key="user")

    st.markdown("---")
    st.write("📊 Features:")
    st.write("✔ Spam Detection")
    st.write("✔ AI Confidence Score")
    st.write("✔ History Tracking")

    st.markdown("---")
    st.info("AI SaaS Portfolio Project")

# =========================
# HEADER
# =========================
st.markdown(
    f"<h1 style='text-align:center;'>📧 SpamShield AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Enterprise Email Protection System</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# INPUT SECTION (CENTER CARD)
# =========================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    text = st.text_area("✉️ Enter Email Content", height=180)

    if st.button("🚀 Analyze Email"):

        if text.strip() == "":
            st.warning("Please enter email text")
        else:
            data = vectorizer.transform([text])
            pred = model.predict(data)[0]

            prob = model.predict_proba(data)[0]
            confidence = np.max(prob) * 100

            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            result = {
                "user": st.session_state.user,
                "text": text,
                "prediction": pred,
                "confidence": confidence,
                "time": time_now
            }

            st.session_state.history.append(result)

            if pred == 1:
                st.error(f"🚫 SPAM DETECTED ({confidence:.2f}%)")
            else:
                st.success(f"✅ NOT SPAM ({confidence:.2f}%)")

# =========================
# ANALYTICS DASHBOARD
# =========================
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

if len(st.session_state.history) > 0:

    df = pd.DataFrame(st.session_state.history)

    spam_count = len(df[df["prediction"] == 1])
    ham_count = len(df[df["prediction"] == 0])

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Checks", len(df))
    col2.metric("Spam Detected", spam_count)
    col3.metric("Safe Emails", ham_count)

# =========================
# HISTORY TABLE
# =========================
st.markdown("---")
st.subheader("📜 Activity History")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df[::-1])
else:
    st.info("No activity yet")

# =========================
# DOWNLOAD REPORT
# =========================
if len(st.session_state.history) > 0:
    csv = pd.DataFrame(st.session_state.history).to_csv(index=False)

    st.download_button(
        "⬇ Download Report",
        csv,
        "spamshield_report.csv",
        "text/csv"
    )