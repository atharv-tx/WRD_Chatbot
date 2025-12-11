import streamlit as st
import json
import os
import requests
import re

# SAFE PDF reader
try:
    import pdfplumber
except:
    pdfplumber = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# ============================================================
# 🧠 CHAT HISTORY + PDF SOURCE MEMORY
# ============================================================

def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "last_pdf_sources" not in st.session_state:
        st.session_state.last_pdf_sources = []

    if "meta_mode" not in st.session_state:
        st.session_state.meta_mode = False


def add_message(role, message):
    st.session_state.chat_history.append({"role": role, "message": message})


def get_history_for_llm():
    # Only last 6 messages → improves quality
    text = ""
    for m in st.session_state.chat_history[-6:]:
        speaker = "User" if m["role"] == "user" else "Assistant"
        text += f"{speaker}: {m['message']}\n"
    return text


def clear_history():
    st.session_state.chat_history = []
    st.session_state.last_pdf_sources = []
    st.session_state.meta_mode = False



# ============================================================
# 🌐 LANGUAGES
# ============================================================

LANGUAGES = {
    "हिंदी": {
        "meta": "यह चैटबॉट WRD के वास्तविक दस्तावेज़ों को स्कैन करके RAG तकनीक से उत्तर देता है।",
        "title": "जल संसाधन विभाग AI Chatbot",
        "desc": "यह चैटबॉट WRD दस्तावेज़ों और आपके PDF से उत्तर देता है।",
        "query": "अपना सवाल लिखिए",
        "button": "उत्तर प्राप्त करें",
        "thinking": "उत्तर तैयार किया जा रहा है...",
        "answer": "चैटबॉट का उत्तर:",
        "pdf": "उपयोग किए गए WRD PDF:",
        "download": "PDF डाउनलोड करें",
        "upload": "PDF अपलोड करें",
        "pdf_override": "उत्तर आपके अपलोड किए गए PDF पर आधारित है।",
        "info": "यह प्रणाली केवल मार्गदर्शन हेतु है।",
    },

    "English": {
        "meta": "This chatbot uses WRD documents & RAG (Retrieval-Augmented Generation) to provide official information.",
        "title": "WRD Chhattisgarh – AI Chatbot",
        "desc": "This chatbot answers using WRD data or your PDF.",
        "query": "Ask your question",
        "button": "Get Answer",
        "thinking": "Generating answer...",
        "answer": "Chatbot Answer:",
        "pdf": "Used WRD PDFs:",
        "download": "Download PDF",
        "upload": "Upload PDF",
        "pdf_override": "Answer based on your uploaded PDF.",
        "info": "This system is for guidance only.",
    },

    "Hinglish": {
        "meta": "Yeh chatbot WRD documents ko RAG ke through analyze karke exact info deta hai.",
        "title": "WRD Chhattisgarh – AI Chatbot",
        "desc": "Ye chatbot WRD ya uploaded PDF se answer deta hai.",
        "query": "Apna sawaal likhiye",
        "button": "Answer Pao",
        "thinking": "Answer tayyar ho raha hai...",
        "answer": "Chatbot Ka Answer:",
        "pdf": "Use huye WRD PDF:",
        "download": "PDF Download",
        "upload": "PDF Upload Karein",
        "pdf_override": "Answer uploaded PDF se liya gaya hai.",
        "info": "Ye system sirf guidance ke liye hai.",
    }
}



# ============================================================
# 🤖 META-QUESTION + PDF-CHECK DETECTION
# ============================================================

META_QUESTIONS = [
    r"what is this chatbot",
    r"what can you do",
    r"who are you",
    r"your purpose",
    r"kaise kaam",
    r"tum kya",
    r"bot kya",
    r"chatbot",
    r"how .* work",
]

META_FOLLOWUP = [
    r"more detail",
    r"detail",
    r"explain",
    r"continue",
    r"aur",
]


WRD_KEYWORDS = [
    "irrigation", "water", "borewell", "dam", "pipeline",
    "canal", "scheme", "wrd", "chhattisgarh", "ground water",
    "act", "permission"
]


# PDF Query detection
PDF_QUERY_PATTERNS = [
    r"which pdf",
    r"list pdf",
    r"which document",
    r"source pdf",
    r"pdf used",
    r"kis pdf",
]

def is_pdf_request(q):
    q = q.lower()
    return any(re.search(p, q) for p in PDF_QUERY_PATTERNS)


def is_meta_question(q):
    q = q.lower()

    # If WRD keywords found → NOT meta
    if any(w in q for w in WRD_KEYWORDS):
        return False

    # If already in meta mode → follow-up continuation
    if st.session_state.meta_mode:
        if any(re.search(p, q) for p in META_FOLLOWUP):
            return True

    # Fresh meta-question
    return any(re.search(p, q) for p in META_QUESTIONS)



# ============================================================
# 📚 LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_resource
def load_kb_and_vectorizer():
    with open("wrd_kb.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts = [f"{d['title']}\n\n{d['text']}" for d in docs]
    meta = [{"title": d["title"], "url": d["url"], "type": d["type"]} for d in docs]

    vec = TfidfVectorizer()
    matrix = vec.fit_transform(texts)

    return docs, meta, vec, matrix


def retrieve_context(query, vectorizer, matrix, docs, meta, top_k):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idxs = sims.argsort()[::-1][:top_k]

    chunks, pdf_list = [], []

    for i in idxs:
        chunks.append(docs[i]["text"][:900])
        if meta[i]["type"] == "pdf":
            pdf_list.append(meta[i])

    return "\n\n----\n\n".join(chunks), pdf_list



# ============================================================
# 🤖 GROQ LLM CALL
# ============================================================

def ask_llm_cloud(query, context, history, lang):
    key = st.secrets["GROQ_API_KEY"]

    final_prompt = f"""
You are WRD Assistant.

RULES:
- For WRD factual questions → use ONLY the context below.
- Chat history is ONLY for tone continuity, NOT facts.
- Give long, accurate, step-by-step answers.

Chat History:
{history}

RAG Context:
{context}

User Question:
{query}
"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are WRD expert assistant."},
            {"role": "user", "content": final_prompt},
        ],
        "temperature": 0.15
    }

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload
    )

    data = res.json()
    return data["choices"][0]["message"]["content"]



# ============================================================
# 🟦 UI (unchanged)
# ============================================================

st.set_page_config(page_title="WRD AI Chatbot", layout="centered")
init_chat_history()

lang = st.selectbox("Select Language / भाषा चुनें", list(LANGUAGES.keys()))
ui = LANGUAGES[lang]

st.title(ui["title"])
st.markdown(ui["desc"])

uploaded_pdf = st.file_uploader(ui["upload"], type=["pdf"])

docs, meta, vectorizer, matrix = load_kb_and_vectorizer()

query = st.text_area(ui["query"])
top_k = st.slider("Top Documents", 1, 5, 3)

pdf_sources = []



# ============================================================
# 🚀 MAIN BUTTON LOGIC
# ============================================================

if st.button(ui["button"]):

    user_q = query.strip()
    history = get_history_for_llm()

    # 1️⃣ If user asks "WHICH PDF DID YOU USE?"
    if is_pdf_request(user_q):
        if st.session_state.last_pdf_sources:
            ans = "📄 PDFs used in last answer:\n\n"
            for p in st.session_state.last_pdf_sources:
                ans += f"- **{p['title']}** → {p['url']}\n"
        else:
            ans = "❗ No PDF was used for the previous answer."

        add_message("user", user_q)
        add_message("assistant", ans)
        st.stop()

    # 2️⃣ META question?
    if is_meta_question(user_q):
        st.session_state.meta_mode = True
        ans = ui["meta"]
        add_message("user", user_q)
        add_message("assistant", ans)
        st.stop()

    # Normal WRD question → turn meta_mode off
    st.session_state.meta_mode = False

    # 3️⃣ Handle PDF override
    if uploaded_pdf:
        context = pdfplumber.open(uploaded_pdf).pages[0].extract_text()[:5000]
        pdf_sources = []
        st.info(ui["pdf_override"])

    else:
        context, pdf_sources = retrieve_context(
            user_q, vectorizer, matrix, docs, meta, top_k
        )

    with st.spinner(ui["thinking"]):
        ans = ask_llm_cloud(user_q, context, history, lang)

    # Save which PDFs were used
    st.session_state.last_pdf_sources = pdf_sources

    add_message("user", user_q)
    add_message("assistant", ans)



# ============================================================
# 💬 CHAT HISTORY
# ============================================================

st.subheader(ui["answer"])
for m in st.session_state.chat_history:
    speaker = "🧑" if m["role"] == "user" else "🤖"
    st.markdown(f"**{speaker} {m['message']}**")



# ============================================================
# 📄 PDF Sources (for WRD)
# ============================================================

if st.session_state.last_pdf_sources:
    st.subheader(ui["pdf"])
    for p in st.session_state.last_pdf_sources:
        st.markdown(f"📄 **{p['title']}** — [{ui['download']}]({p['url']})")



# ============================================================
# CLEAR CHAT
# ============================================================

if st.button("🗑 Clear Chat"):
    clear_history()
    st.success("Chat Cleared!")

st.info(ui["info"])
