import streamlit as st
import json
import os
import requests
import io
import re

try:
    import pdfplumber
except:
    pdfplumber = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# CONFIG
# ----------------------------
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

PDFS_FOLDERS = ["pdfs", "wrd_pdfs"]   # <--- NOW BOTH FOLDERS INCLUDED
KB_FILE = "wrd_kb.json"

MAX_CONTEXT = 4000
HISTORY_LIMIT = 6


# ----------------------------
# Languages UI
# ----------------------------
LANG = {
    "English": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Answers using WRD documents & your PDFs.",
        "query": "Enter your question",
        "button": "Get Answer",
        "thinking": "Generating answer...",
        "answer": "Chatbot Answer:",
        "pdf": "Referenced PDFs:",
        "download": "Download PDF",
        "upload": "Upload PDF",
        "meta": "I am a WRD chatbot that uses RAG to read WRD PDFs and answer queries.",
        "info": "This system is for guidance only.",
        "pdf_read_error": "PDF reading is not available."
    },
    "हिंदी": {
        "title": "💧 जल संसाधन विभाग छत्तीसगढ़ – एआई चैटबॉट",
        "desc": "WRD दस्तावेज़ों और PDF से उत्तर प्रदान करता है।",
        "query": "अपना सवाल लिखें",
        "button": "उत्तर प्राप्त करें",
        "thinking": "उत्तर तैयार किया जा रहा है...",
        "answer": "चैटबॉट का उत्तर:",
        "pdf": "संदर्भित PDF:",
        "download": "PDF डाउनलोड",
        "upload": "PDF अपलोड करें",
        "meta": "मैं एक WRD चैटबॉट हूँ जो WRD PDF पढ़कर RAG से उत्तर देता हूँ।",
        "info": "यह प्रणाली केवल मार्गदर्शन हेतु है।",
        "pdf_read_error": "PDF पढ़ना उपलब्ध नहीं है।"
    }
}


# ----------------------------
# History
# ----------------------------
def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_used_pdfs" not in st.session_state:
        st.session_state.last_used_pdfs = []


def add_message(role, text):
    st.session_state.history.append({"role": role, "text": text})


def get_tone_history():
    out = ""
    for h in st.session_state.history[-HISTORY_LIMIT:]:
        speaker = "User" if h["role"] == "user" else "Assistant"
        out += f"{speaker}: {h['text']}\n"
    return out


# ----------------------------
# META QUESTION
# ----------------------------
META_PATTERNS = [
    r"who are you", r"what can you do", r"what is this chatbot",
    r"your purpose", r"bot kya", r"kaise kaam"
]

def is_meta(q):
    q = q.lower()
    domain = ["water", "wrd", "irrigation", "canal", "dam", "borewell"]
    if any(x in q for x in domain):
        return False
    return any(re.search(p, q) for p in META_PATTERNS)


# ----------------------------
# LOAD KB + PDFs
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_kb_and_pdfs():
    import time
    start = time.time()

    docs = []
    meta = []

    MAX_PDF_CHARS = 4000     # small slice → speeds up processing a LOT
    MAX_DOCS = 50            # optional limit (remove if needed)

    # Load KB JSON
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        for d in data:
            txt = d["text"][:MAX_PDF_CHARS]
            docs.append(txt)
            meta.append({"title": d["title"], "url": d["url"], "type": d["type"]})

    # Folders to scan
    folders = ["pdfs", "wrd_pdfs"]

    # Load PDFs (FAST MODE)
    if pdfplumber:
        for folder in folders:
            if os.path.isdir(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith(".pdf"):

                        if len(docs) >= MAX_DOCS:
                            break

                        path = os.path.join(folder, fname)
                        try:
                            # Only read first 3 pages for speed
                            with pdfplumber.open(path) as pdf:
                                pages = []
                                for i, p in enumerate(pdf.pages):
                                    if i >= 3:
                                        break
                                    pages.append(p.extract_text() or "")

                            text = "\n".join(pages)[:MAX_PDF_CHARS]

                            docs.append(text)
                            meta.append({"title": fname, "url": path, "type": "pdf"})

                        except Exception as e:
                            print("PDF read error:", path, e)

    # fallback
    if not docs:
        docs = ["No WRD documents found."]
        meta = [{"title": "None", "url": "", "type": "none"}]

    # Vectorizer (FAST MODE)
    vectorizer = TfidfVectorizer(
        max_features=5000,          # limit vocabulary → much faster
        stop_words="english"        # reduces tokens → faster vectorization
    )

    matrix = vectorizer.fit_transform(docs)

    end = time.time()
    print(f"✔ Loaded KB + PDFs in {end-start:.2f} seconds")

    return docs, meta, vectorizer, matrix



# ----------------------------
# RETRIEVAL
# ----------------------------
def retrieve(query, docs, meta, vect, matrix, k):
    q_vec = vect.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idxs = sims.argsort()[::-1][:k]

    chunks = []
    used = []

    for i in idxs:
        chunks.append(docs[i][:MAX_CONTEXT])
        if meta[i]["type"] == "pdf":
            used.append(meta[i])

    return "\n\n----\n\n".join(chunks), used


# ----------------------------
# GROQ API CALL
# ----------------------------
def ask_groq(q, context, history):
    if "GROQ_API_KEY" not in st.secrets:
        return "❌ Missing GROQ_API_KEY"

    final = f"""
You are WRD Assistant. Use context for factual WRD answers.
If WRD info missing, give general guidance with disclaimer.

History:
{history}

Context:
{context}

User question:
{q}
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are WRD expert assistant."},
            {"role": "user", "content": final},
        ],
        "temperature": 0.15,
    }

    r = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"},
        json=payload
    )

    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except:
        return str(j)


# ----------------------------
# UI START
# ----------------------------
st.set_page_config(page_title="WRD AI Chatbot")
init_history()

language = st.selectbox("Language", list(LANG.keys()))
UI = LANG[language]

st.title(UI["title"])
st.write(UI["desc"])

uploaded_pdf = st.file_uploader(UI["upload"], type=["pdf"])
query = st.text_area(UI["query"])
top_k = st.slider("Documents to retrieve", 1, 5, 3)

docs, meta, vect, matrix = load_kb_and_pdfs()

# ----------------------------
# PROCESS QUERY
# ----------------------------
if st.button(UI["button"]):
    q = query.strip()
    add_message("user", q)

    if is_meta(q):
        ans = UI["meta"]
        add_message("assistant", ans)

    else:
        if uploaded_pdf and pdfplumber:
            with pdfplumber.open(uploaded_pdf) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            context = "\n".join(pages)[:MAX_CONTEXT]
            used = [{"title": uploaded_pdf.name, "url": "uploaded", "type": "pdf"}]
        else:
            context, used = retrieve(q, docs, meta, vect, matrix, top_k)

        st.session_state["last_used_pdfs"] = used

        with st.spinner(UI["thinking"]):
            ans = ask_groq(q, context, get_tone_history())

        add_message("assistant", ans)


# ----------------------------
# SHOW CHAT
# ----------------------------
st.subheader(UI["answer"])

for h in st.session_state.history:
    if h["role"] == "user":
        st.markdown(f"**🧑 User:** {h['text']}")
    else:
        st.markdown(f"**🤖 Bot:** {h['text']}")


# ----------------------------
# SHOW REFERENCED PDFs
# ----------------------------
if st.session_state["last_used_pdfs"]:
    st.subheader(UI["pdf"])
    for pdf in st.session_state["last_used_pdfs"]:
        if pdf["url"] == "uploaded":
            st.markdown(f"📄 **{pdf['title']}** (User uploaded)")
        else:
            st.markdown(f"📄 **{pdf['title']}** — Located at: `{pdf['url']}`")


# ----------------------------
# CLEAR CHAT
# ----------------------------
if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.last_used_pdfs = []
    st.success("Chat cleared.")

st.info(UI["info"])
