import streamlit as st
import json
import os
import requests

# ✅ SAFE pdfplumber fallback
try:
    import pdfplumber
except:
    pdfplumber = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# 🌐 LANGUAGE CONFIG
# -------------------------

LANGUAGES = {
    "हिंदी": {
        "title": "💧 जल संसाधन विभाग छत्तीसगढ़ – एआई चैटबॉट",
        "desc": "यह chatbot WRD दस्तावेज़ों और आपके PDF से उत्तर देता है।",
        "query": "✍️ अपना सवाल लिखिए",
        "button": "✅ उत्तर प्राप्त करें",
        "search": "🔎 जानकारी खोजी जा रही है...",
        "thinking": "🤖 उत्तर तैयार किया जा रहा है...",
        "answer": "🤖 चैटबॉट का उत्तर:",
        "pdf": "📄 उपयोग किए गए WRD PDF दस्तावेज़:",
        "download": "⬇️ PDF डाउनलोड करें",
        "upload": "➕ अपना PDF अपलोड करें (वैकल्पिक)",
        "pdf_override": "✅ उत्तर आपके अपलोड किए गए PDF से तैयार किया गया है।",
        "info": "ℹ️ यह प्रणाली केवल मार्गदर्शन हेतु है।"
    },
    "English": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "This chatbot answers from WRD documents and your uploaded PDF.",
        "query": "✍️ Enter your question",
        "button": "✅ Get Answer",
        "search": "🔎 Searching information...",
        "thinking": "🤖 Generating answer...",
        "answer": "🤖 Chatbot Answer:",
        "pdf": "📄 Used WRD PDF Documents:",
        "download": "⬇️ Download PDF",
        "upload": "➕ Upload your own PDF (optional)",
        "pdf_override": "✅ Answer is generated from your uploaded PDF.",
        "info": "ℹ️ This system is for guidance only."
    },
    "Hinglish": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Ye chatbot WRD docs aur aapke PDF se answer deta hai.",
        "query": "✍️ Apna sawaal likhiye",
        "button": "✅ Answer Pao",
        "search": "🔎 Info dhoondi ja rahi hai...",
        "thinking": "🤖 Answer banaya ja raha hai...",
        "answer": "🤖 Chatbot Answer:",
        "pdf": "📄 Use hue WRD PDF:",
        "download": "⬇️ PDF Download",
        "upload": "➕ Apna PDF upload karein",
        "pdf_override": "✅ Answer uploaded PDF se diya gaya hai.",
        "info": "ℹ️ Ye system sirf guidance ke liye hai."
    }
}


# -------------------------
# 1. Load WRD Knowledge Base
# -------------------------

@st.cache_resource
def load_kb_and_vectorizer():
    with open("wrd_kb.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = []
    meta = []

    for d in docs:
        combined = f"{d.get('title', '')}\n\n{d.get('text', '')}"
        texts.append(combined)
        meta.append({
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "type": d.get("type", "")
        })

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(texts)

    return docs, meta, vectorizer, doc_matrix


def retrieve_context(query, vectorizer, doc_matrix, docs, meta, top_k=3):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, doc_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    chunks = []
    pdf_sources = []

    for idx in top_idx:
        chunks.append(docs[idx]["text"][:900])
        if meta[idx]["type"].lower() == "pdf":
            pdf_sources.append(meta[idx])

    return "\n\n----\n\n".join(chunks), pdf_sources


# -------------------------
# 2. PDF READER
# -------------------------

def read_uploaded_pdf(uploaded_file):
    if pdfplumber is None:
        return "PDF reading not supported on this server."

    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                full_text += t + "\n"
    return full_text[:4000]


# -------------------------
# 3. CLOUD LLM (GROQ)
# -------------------------

def ask_llm_cloud(query, context, selected_lang):
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    prompt = f"""
You are a government information assistant.

Answer in this language: {selected_lang}
Use ONLY the given context.
Give a long, detailed, step-by-step answer.

Context:
{context}

Question:
{query}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    return response.json()["choices"][0]["message"]["content"]


# -------------------------
# 4. STREAMLIT UI
# -------------------------

st.set_page_config(page_title="WRD AI Chatbot", layout="centered")

selected_lang = st.selectbox("🌐 Select Language / भाषा चुनें", list(LANGUAGES.keys()))
ui = LANGUAGES[selected_lang]

st.title(ui["title"])
st.markdown(ui["desc"])

uploaded_pdf = st.file_uploader(ui["upload"], type=["pdf"])

docs, meta, vectorizer, doc_matrix = load_kb_and_vectorizer()

query = st.text_area(ui["query"], height=140)
top_k = st.slider("📄 Top Documents", 1, 5, 3)

if st.button(ui["button"]):
    if uploaded_pdf:
        context = read_uploaded_pdf(uploaded_pdf)
        pdf_sources = []
        st.info(ui["pdf_override"])
    else:
        context, pdf_sources = retrieve_context(
            query, vectorizer, doc_matrix, docs, meta, top_k
        )

    with st.spinner(ui["thinking"]):
        answer = ask_llm_cloud(query, context, selected_lang)

    st.subheader(ui["answer"])
    st.success(answer)

    if not uploaded_pdf:
        st.subheader(ui["pdf"])
        for s in pdf_sources:
            st.markdown(f"✅ **{s['title']}**")
            st.markdown(f"[{ui['download']}]({s['url']})")

st.info(ui["info"])
