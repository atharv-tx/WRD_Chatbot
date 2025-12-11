import streamlit as st
import json
import os
import requests

# SAFE pdfplumber fallback
try:
    import pdfplumber
except:
    pdfplumber = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------------------------
# GOVT PORTAL UI THEME (INLINE CSS)
# -------------------------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #f1f6fd;
}

/* Header */
.govt-header {
    background-color: #0b3d91;
    padding: 18px;
    color: white;
    border-radius: 8px;
    display: flex;
    align-items: center;
}

.govt-header img {
    height: 55px;
    margin-right: 18px;
}

.sec-card {
    background-color: white;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.10);
    margin-bottom: 20px;
}

.chat-bubble {
    background: #e9f1ff;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #0b3d91;
    font-size: 18px;
}

.pdf-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border-left: 5px solid #0b3d91;
    margin-bottom: 10px;
}

.upload-box {
    border: 2px dashed #0b3d91;
    padding: 20px;
    border-radius: 10px;
    background: #f8fbff;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# LANGUAGE SETTINGS
# -------------------------------------------------------------------

LANGUAGES = {
    "English": {
        "query": "✍️ Enter your question",
        "button": "Get Answer",
        "upload": "Upload your own PDF (optional)",
        "search": "Searching WRD documents...",
        "thinking": "Generating Answer...",
        "answer": "Chatbot Answer",
        "pdf": "WRD Documents Used",
        "download": "Download PDF",
        "pdf_override": "Answer taken ONLY from your uploaded PDF.",
    },
    "हिंदी": {
        "query": "✍️ अपना सवाल लिखिए",
        "button": "उत्तर प्राप्त करें",
        "upload": "अपना PDF अपलोड करें (वैकल्पिक)",
        "search": "WRD दस्तावेज़ खोजे जा रहे हैं...",
        "thinking": "उत्तर तैयार किया जा रहा है...",
        "answer": "चैटबॉट का उत्तर",
        "pdf": "उपयोग किए गए WRD PDF",
        "download": "PDF डाउनलोड करें",
        "pdf_override": "उत्तर केवल आपके अपलोड किए गए PDF से बनाया गया है।",
    }
}


# -------------------------------------------------------------------
# LOAD KNOWLEDGE BASE
# -------------------------------------------------------------------

@st.cache_resource
def load_kb_and_vectorizer():
    with open("wrd_kb.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = []
    meta = []
    for d in docs:
        combined = f"{d.get('title', '')}\n{d.get('text', '')}"
        texts.append(combined)
        meta.append(d)

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(texts)

    return docs, meta, vectorizer, doc_matrix


def retrieve_context(query, vectorizer, doc_matrix, docs, meta, top_k=3):
    sims = cosine_similarity(vectorizer.transform([query]), doc_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    chunks, pdf_sources = [], []
    for idx in top_idx:
        chunks.append(docs[idx]["text"][:900])
        if meta[idx]["type"].lower() == "pdf":
            pdf_sources.append(meta[idx])

    return "\n----\n".join(chunks), pdf_sources


# -------------------------------------------------------------------
# PDF READER
# -------------------------------------------------------------------

def read_uploaded_pdf(uploaded_file):
    if pdfplumber is None:
        return "PDF reading is not supported on server."
    full = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                full += t + "\n"
    return full[:4000]


# -------------------------------------------------------------------
# LLM (GROQ Cloud)
# -------------------------------------------------------------------

def ask_llm_cloud(query, context, lang):
    if "GROQ_API_KEY" not in st.secrets:
        return "❌ GROQ_API_KEY missing in cloud secrets."

    key = st.secrets["GROQ_API_KEY"]

    prompt = f"""
Answer ONLY in: {lang}
Use ONLY this context:

{context}

Give a very detailed, long and structured answer.
"""

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
    )

    data = res.json()
    if "choices" not in data:
        return f"❌ Invalid Groq Response: {data}"

    return data["choices"][0]["message"]["content"]


# -------------------------------------------------------------------
# GOVERNMENT HEADER
# -------------------------------------------------------------------

st.markdown("""
<div class="govt-header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/800px-Emblem_of_India.svg.png">
    <h2>Water Resources Department, Government of Chhattisgarh</h2>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# MAIN UI LAYOUT: TWO COLUMNS (GOVT PORTAL STYLE)
# -------------------------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='sec-card'>", unsafe_allow_html=True)

    lang = st.selectbox("🌐 Language / भाषा", list(LANGUAGES.keys()))
    ui = LANGUAGES[lang]

    query = st.text_area(ui["query"], height=150)

    uploaded_pdf = st.file_uploader(ui["upload"], type=["pdf"])

    top_k = st.slider("📄 Top Documents", 1, 5, 3)

    if st.button(ui["button"]):
        if uploaded_pdf:
            context = read_uploaded_pdf(uploaded_pdf)
            pdf_used = []
            st.info(ui["pdf_override"])
        else:
            context, pdf_used = retrieve_context(query, *load_kb_and_vectorizer(), top_k)

        with st.spinner(ui["thinking"]):
            answer = ask_llm_cloud(query, context, lang)

        st.session_state["answer"] = answer
        st.session_state["pdf_used"] = pdf_used

    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown("<div class='sec-card'>", unsafe_allow_html=True)

    if "answer" in st.session_state:
        st.subheader(ui["answer"])
        st.markdown(f"<div class='chat-bubble'>{st.session_state['answer']}</div>", unsafe_allow_html=True)

        if st.session_state["pdf_used"]:
            st.subheader(ui["pdf"])
            for p in st.session_state["pdf_used"]:
                st.markdown(f"""
                <div class="pdf-card">
                    <b>{p['title']}</b><br>
                    <a href="{p['url']}" target="_blank">{ui['download']}</a>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
