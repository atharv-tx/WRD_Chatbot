import streamlit as st
import json
import os
import requests

# ✅ SAFE pdfplumber fallback for Streamlit Cloud
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
        "desc": "यह चैटबॉट WRD के वास्तविक दस्तावेज़ों और आपके अपलोड किए गए PDF से उत्तर देता है।",
        "query": "✍️ अपना सवाल लिखिए",
        "button": "✅ उत्तर प्राप्त करें",
        "search": "🔎 दस्तावेज़ों से जानकारी खोजी जा रही है...",
        "thinking": "🤖 उत्तर तैयार किया जा रहा है...",
        "answer": "🤖 चैटबॉट का उत्तर:",
        "pdf": "📄 उपयोग किए गए WRD PDF दस्तावेज़:",
        "download": "⬇️ PDF डाउनलोड करें",
        "upload": "➕ अपना PDF अपलोड करें (अगर आप WRD के अलावा उसी PDF से उत्तर चाहते हैं)",
        "pdf_override": "✅ उत्तर केवल आपके अपलोड किए गए PDF से तैयार किया गया है।",
        "info": "ℹ️ यह प्रणाली केवल मार्गदर्शन हेतु है।"
    },
    "English": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "This chatbot answers using official WRD documents and your uploaded PDF.",
        "query": "✍️ Enter your question",
        "button": "✅ Get Answer",
        "search": "🔎 Searching documents...",
        "thinking": "🤖 Generating answer...",
        "answer": "🤖 Chatbot Answer:",
        "pdf": "📄 Used WRD PDF Documents:",
        "download": "⬇️ Download PDF",
        "upload": "➕ Upload your own PDF (to override WRD data)",
        "pdf_override": "✅ Answer is generated ONLY from your uploaded PDF.",
        "info": "ℹ️ This system is for guidance only."
    },
    "Hinglish": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Ye chatbot WRD ke documents aur aapke upload PDF se answer deta hai.",
        "query": "✍️ Apna sawaal likhiye",
        "button": "✅ Answer Pao",
        "search": "🔎 Documents se info dhoondi ja rahi hai...",
        "thinking": "🤖 Answer banaya ja raha hai...",
        "answer": "🤖 Chatbot Ka Answer:",
        "pdf": "📄 Use hue WRD PDF Documents:",
        "download": "⬇️ PDF Download Karein",
        "upload": "➕ Apna PDF upload karein (WRD ko ignore karne ke liye)",
        "pdf_override": "✅ Answer sirf aapke uploaded PDF se banaya gaya hai.",
        "info": "ℹ️ Ye system sirf guidance ke liye hai."
    }
}


# -------------------------
# 1. Load WRD Knowledge Base
# -------------------------

@st.cache_resource
def load_kb_and_vectorizer():
    if not os.path.exists("wrd_kb.json"):
        raise FileNotFoundError("❌ wrd_kb.json नहीं मिला। पहले fetch_wrd_data.py चलाएँ।")

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
        d = docs[idx]
        chunks.append(d["text"][:800])

        if meta[idx]["type"].lower() == "pdf":
            pdf_sources.append(meta[idx])

    return "\n\n----\n\n".join(chunks), pdf_sources


# -------------------------
# 2. USER UPLOADED PDF READER (SAFE MODE)
# -------------------------

def read_uploaded_pdf(uploaded_file):
    if pdfplumber is None:
        return "⚠️ PDF पढ़ने की सुविधा इस सर्वर पर अस्थायी रूप से उपलब्ध नहीं है। कृपया केवल WRD डेटा से प्रश्न पूछें।"

    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    return full_text[:3500]


# -------------------------
# 3. Ollama (LONG + AUTO CONTINUE)
# -------------------------

def ask_llm_ollama(query, context, selected_lang):
    system_prompt = f"""
You are a WRD Chhattisgarh assistant.
Give a very long, detailed and informative answer ONLY in this language: {selected_lang}.
Use ONLY the given context.
Explain procedures, steps, documents, eligibility, and rules in detail.
If info is not present, clearly say it is unavailable.
"""

    def generate_once(prompt):
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.1",
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 700,
                    "temperature": 0.15,
                    "top_p": 0.95
                },
            },
            timeout=120,
            stream=True
        )

        final_text = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                if "response" in data:
                    final_text += data["response"]
                if data.get("done"):
                    break
        return final_text.strip()

    full_prompt = f"""
{system_prompt}

Context:
{context}

Question:
{query}
"""

    answer = generate_once(full_prompt)

    if len(answer) < 900:
        continuation = generate_once("Continue the same answer in full detail:")
        answer += "\n" + continuation

    return answer.strip()


# -------------------------
# 4. Streamlit UI
# -------------------------

st.set_page_config(page_title="WRD AI Chatbot", layout="centered")

selected_lang = st.selectbox("🌐 Select Language / भाषा चुनें", list(LANGUAGES.keys()))
ui = LANGUAGES[selected_lang]

st.title(ui["title"])
st.markdown(ui["desc"])

uploaded_pdf = st.file_uploader(ui["upload"], type=["pdf"])

try:
    docs, meta, vectorizer, doc_matrix = load_kb_and_vectorizer()
except Exception as e:
    st.error(str(e))
    st.stop()

query = st.text_area(ui["query"], height=140)
top_k = st.slider("📄 Top Documents", 1, 5, 3)

if st.button(ui["button"]):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner(ui["search"]):

            if uploaded_pdf is not None:
                context = read_uploaded_pdf(uploaded_pdf)
                pdf_sources = []
                st.info(ui["pdf_override"])
            else:
                context, pdf_sources = retrieve_context(
                    query, vectorizer, doc_matrix, docs, meta, top_k
                )

        with st.spinner(ui["thinking"]):
            answer = ask_llm_ollama(query, context, selected_lang)

        st.subheader(ui["answer"])
        st.success(answer)

        if uploaded_pdf is None:
            st.subheader(ui["pdf"])
            if pdf_sources:
                for s in pdf_sources:
                    st.markdown(f"✅ **{s['title']}**")
                    st.markdown(f"🔗 {s['url']}")
                    st.markdown(f"[{ui['download']}]({s['url']})")
                    st.markdown("---")
            else:
                st.info("इस उत्तर के लिए कोई WRD PDF उपयोग में नहीं लिया गया।")

st.info(ui["info"])
