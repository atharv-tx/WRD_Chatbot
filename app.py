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
        "desc": "यह चैटबॉट WRD दस्तावेज़ों और आपके PDF से उत्तर देता है।",
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
        "desc": "This chatbot answers using WRD data or your uploaded PDF.",
        "query": "✍️ Enter your question",
        "button": "✅ Get Answer",
        "search": "🔎 Searching documents...",
        "thinking": "🤖 Generating answer...",
        "answer": "🤖 Chatbot Answer:",
        "pdf": "📄 Used WRD PDF Documents:",
        "download": "⬇️ Download PDF",
        "upload": "➕ Upload your PDF (optional)",
        "pdf_override": "✅ Answer generated from your uploaded PDF.",
        "info": "ℹ️ This system is for guidance only."
    },
    "Hinglish": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Ye chatbot WRD aur uploaded PDF se answer deta hai.",
        "query": "✍️ Apna sawaal likhiye",
        "button": "✅ Answer Pao",
        "search": "🔎 Documents search ho rahe hain...",
        "thinking": "🤖 Answer banaya ja raha hai...",
        "answer": "🤖 Chatbot ka Answer:",
        "pdf": "📄 Use hue WRD PDF:",
        "download": "⬇️ PDF Download",
        "upload": "➕ Apna PDF Upload Karein",
        "pdf_override": "✅ Answer sirf uploaded PDF se diya gaya hai.",
        "info": "ℹ️ Ye system sirf guidance ke liye hai."
    }
}

# -------------------------
# 1. Load WRD Knowledge Base
# -------------------------

@st.cache_resource
def load_kb_and_vectorizer():
    if not os.path.exists("wrd_kb.json"):
        st.error("❌ wrd_kb.json नहीं मिला।")
        st.stop()

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
        return "❌ PDF reader supported नहीं है।"

    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    return text[:4000]


# -------------------------
# 3. GROQ CLOUD LLM (ONLY)
# -------------------------

def ask_llm_cloud(query, context, selected_lang):
    try:
        if "GROQ_API_KEY" not in st.secrets:
            return "❌ GROQ_API_KEY Streamlit Secrets में नहीं मिला।"

        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

        prompt = f"""
You are an official WRD information assistant.
Answer strictly in this language: {selected_lang}.
Use ONLY the given context.
Provide a long, detailed, step-by-step answer.

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
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful WRD assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        data = response.json()

        if response.status_code != 200:
            return f"❌ Groq API Error {response.status_code}: {data}"

        if "choices" not in data:
            return f"❌ Invalid Groq Response: {data}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"❌ Network Error: {str(e)}"


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
    if not query.strip():
        st.warning("❗ कृपया प्रश्न लिखें।")
    else:
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
                st.markdown(f"📄 **{s['title']}**")
                st.markdown(f"[{ui['download']}]({s['url']})")

st.info(ui["info"])
