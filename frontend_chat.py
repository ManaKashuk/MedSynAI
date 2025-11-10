# ===============================
# 🧬 MedSyn AI — Medical Synonym Assistant (Offline  Mode)
# ===============================

import streamlit as st
import pandas as pd
from PIL import Image
import requests
import os

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="MedSyn AI: Medical Synonym Assistant",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------
# HEADER SECTION
# -------------------------------
logo_path = "logo.png"
try:
    logo = Image.open(logo_path)
    st.image(logo, width=500)
except Exception:
    st.warning("⚠️ Logo not found. Please place 'logo.png' in the same folder.")

st.markdown(
    """
    <p style='font-size: 1.1em; color: #6e7467;'>
    MedSyn AI is a semantic assistant designed to unify medical terminology, enabling fast synonym discovery,
    contextual understanding, and data interoperability across biomedical datasets.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# BACKEND OR  BACKUP
# -------------------------------
API_URL = "http://127.0.0.1:8000"
BACKUP_FILE = "medsyn_backup.csv"

use_backup = False

# Check backend availability
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        st.success("✅ Backend is online and ready")
    else:
        use_backup = True
except Exception:
    st.warning("⚠️ Backend not reachable — using local  backup instead.")
    use_backup = True

# Load  backup
if use_backup:
    if os.path.exists(BACKUP_FILE):
        df = pd.read_(medsyn_backup.csv)
        st.info(f"📁 Loaded offline backup: `{BACKUP_FILE}`")
    else:
        st.error("❌ Backup  not found. Please add `medsyn_backup.xlsx` to the repo.")
        st.stop()

# -------------------------------
# CATEGORY + TERM SELECTION
# -------------------------------
categories = sorted(df["Category"].unique()) if use_backup else [
    "🧬 Cell Processes",
    "🧫 Diseases",
    "🧠 Genes & Proteins",
    "💊 Drug Classes",
    "🏥 Clinical Terms"
]

st.markdown("---")
st.subheader("🔍 Explore Medical Terminology")

selected_category = st.selectbox("Select a category:", categories)

if use_backup:
    terms = df[df["Category"] == selected_category]["Term"].unique().tolist()
else:
    terms = []

selected_term = st.selectbox("Choose a suggested keyword:", [""] + terms)

# -------------------------------
# CHAT & QUERY LOGIC
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Interactive Chat")

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Use manual or dropdown term
prompt = st.chat_input("Enter a medical term or NCIT code...")
if not prompt and selected_term:
    prompt = selected_term

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing term..."):
            try:
                if use_backup:
                    # Look up term in 
                    result_row = df[df["Term"].str.lower() == prompt.lower()]
                    if not result_row.empty:
                        synonyms = result_row.iloc[0]["Synonyms"]
                        definition = result_row.iloc[0]["Definition"]
                        reply = f"### 🧠 **Results for '{prompt}'**\n"
                        reply += f"**Synonyms:** {synonyms}\n\n**Definition:** {definition}"
                    else:
                        reply = f"⚠️ No data found for '{prompt}' in backup file."
                else:
                    # Placeholder for backend
                    reply = f"🔗 Backend mode will query: {API_URL}/... (currently offline)"

            except Exception as e:
                reply = f"❌ Error: {str(e)}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "<center><p style='color:#9e9e9e;'>MedSyn AI © 2025 | Developed by Scientists for Experts 🎯 Built to Unify Medical Terminology Through Semantic Intelligence</p></center>",
    unsafe_allow_html=True
)
