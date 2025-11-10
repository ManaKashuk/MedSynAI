# ===============================
# 🧬 MedSyn AI — Medical Synonym Assistant
# ===============================

import streamlit as st
import requests
from PIL import Image

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="MedSyn AI:Medical Synonym Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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
    <h1 style='font-size: 2em; color: #000000;'>🧬 MedSyn AI is a semantic assistant designed to unify medical terminology, enabling fast synonym discovery,
    contextual understanding, and data interoperability across biomedical datasets.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------
# BACKEND CONFIGURATION
# -------------------------------
API_URL = "http://127.0.0.1:8000"  # change this to your deployed backend URL when live

def query_exact_match_by_code(code):
    res = requests.post(f"{API_URL}/exact_match/by_code", json={"code": code})
    return res.json() if res.status_code == 200 else {"error": res.text}

def query_exact_match_by_term(term):
    res = requests.post(f"{API_URL}/exact_match/by_term", json={"term": term})
    return res.json() if res.status_code == 200 else {"error": res.text}

def query_synonym_by_code(code):
    res = requests.post(f"{API_URL}/synonym/by_code", json={"code": code})
    return res.json() if res.status_code == 200 else {"error": res.text}

def query_synonym_by_term(term):
    res = requests.post(f"{API_URL}/synonym/by_term", json={"term": term})
    return res.json() if res.status_code == 200 else {"error": res.text}

# Optional: for semantic-level results (future extension)
def query_semantic_by_code(code):
    res = requests.post(f"{API_URL}/semantic/by_code", json={"code": code})
    return res.json() if res.status_code == 200 else {"error": res.text}

def query_semantic_by_term(term):
    res = requests.post(f"{API_URL}/semantic/by_term", json={"term": term})
    return res.json() if res.status_code == 200 else {"error": res.text}

# -------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Backend URL:**")
    api_endpoint = st.text_input("API Endpoint", API_URL)
    st.markdown("---")

    st.markdown("**Query Mode**")
    query_type = st.radio(
        "Choose query type:",
        ["Exact Match", "Synonym Search", "Semantic Search"],
        index=1,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("🧬 MedSyn AI v1.0 | © 2025")

# -------------------------------
# MAIN CHAT INTERFACE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 MedSyn AI Interactive Chat")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Enter a medical term or NCIT code..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing term..."):
            try:
                # Determine if the input looks like a code or a term
                if prompt.strip().upper().startswith("C") and prompt[1:].isdigit():
                    # Likely a code
                    if query_type == "Exact Match":
                        result = query_exact_match_by_code(prompt)
                    elif query_type == "Synonym Search":
                        result = query_synonym_by_code(prompt)
                    else:
                        result = query_semantic_by_code(prompt)
                else:
                    # Likely a term
                    if query_type == "Exact Match":
                        result = query_exact_match_by_term(prompt)
                    elif query_type == "Synonym Search":
                        result = query_synonym_by_term(prompt)
                    else:
                        result = query_semantic_by_term(prompt)

                if "error" in result:
                    reply = f"⚠️ Error: {result['error']}"
                else:
                    reply = f"### 🧠 Results for **'{prompt}'**\n"
                    reply += f"```json\n{result}\n```"

            except Exception as e:
                reply = f"❌ Backend connection error: {str(e)}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown(
    "<center><p style='color:#9e9e9e;'>MedSyn AI © 2025 | Built with ❤️ using Streamlit & FastAPI</p></center>",
    unsafe_allow_html=True
)
