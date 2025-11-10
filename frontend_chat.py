# ===============================
# 🧬 MedSyn AI — Medical Synonym Assistant (Offline CSV Mode)
# ===============================

import streamlit as st
import pandas as pd
from PIL import Image
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
    💡 MedSyn AI is a semantic assistant designed to unify medical terminology, enabling fast synonym discovery,
    contextual understanding, and data interoperability across biomedical datasets.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# LOAD LOCAL CSV BACKUP
# -------------------------------
BACKUP_FILE = "medsyn_backup.csv"

if os.path.exists(BACKUP_FILE):
    try:
        df = pd.read_csv(BACKUP_FILE)
        st.success(f"📁 Loaded local backup data from `{BACKUP_FILE}`")
    except Exception as e:
        st.error(f"❌ Failed to load `{BACKUP_FILE}`: {e}")
        st.stop()
else:
    st.error("❌ Backup file not found. Please add `medsyn_backup.csv` to the repository.")
    st.stop()

# -------------------------------
# CATEGORY + TERM SELECTION
# -------------------------------
categories = sorted(df["Category"].unique())

st.markdown("---")
st.subheader("🔍 Explore Medical Terminology")

selected_category = st.selectbox("Select a category:", categories)

terms = df[df["Category"] == selected_category]["Term"].unique().tolist()
selected_term = st.selectbox("Choose a suggested keyword:", [""] + terms)

# -------------------------------
# CHAT & QUERY LOGIC
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Interactive Chat")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Determine input (manual or from dropdown)
prompt = st.chat_input("Enter a medical term or NCIT code...")
if not prompt and selected_term:
    prompt = selected_term

# Chat interaction
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
# Replace the assistant chat message avatar
st.chat_message("assistant", avatar="icon.png").markdown(reply)
        with st.chat_message("assistant"):
        st.markdown(reply)
    
        with st.spinner("Analyzing term..."):
            try:
                result_row = df[df["Term"].str.lower() == prompt.lower()]
                if not result_row.empty:
                    synonyms = result_row.iloc[0]["Synonyms"]
                    definition = result_row.iloc[0]["Definition"]
                    reply = f"### 🧠 **Results for '{prompt}'**\n"
                    reply += f"**Synonyms:** {synonyms}\n\n"
                    reply += f"**Definition:** {definition}"
                else:
                    reply = f"⚠️ No data found for '{prompt}' in the local backup."
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
