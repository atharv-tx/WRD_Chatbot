import streamlit as st
import json
import os
import requests

# SAFE PDF reader
try:
    import pdfplumber
except:
    pdfplumber = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 🔥 CHAT HISTORY MANAGEMENT
# ---------------------------------------------------------

def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role, message):
    st.session_state.chat_history.append({
        "role": role,
        "message": message
    })


def get_history_for_llm():
    text = ""
    for msg in st.session_state.chat_history:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        text += f"{speaker}: {msg['message']}\n"
    return text


def clear_history():
    st.session_state.chat_history = []


# ---------------------------------------------------------
# 🌐 LANGUAGE CONFIG (HINDI + ENGLISH + HINGLISH RESTORED)
# ---------------------------------------------------------

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
        "pdf_override": "📘 उत्तर आपके अपलोड किए गए PDF पर आधारित है।",
        "info": "ℹ️ यह प्रणाली केवल मार्गदर्शन हेतु है।"
    },

    "English": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "This chatbot answers using WRD data or your uploaded PDF.",
        "query": "✍️ Enter your question",
        "button": "✅ Get Answer",
        "search": "🔎 Searching...",
        "thinking": "🤖 Generating...",
        "answer": "🤖 Chatbot Answer:",
        "pdf": "📄 Used WRD PDFs:",
        "download": "⬇️ Download PDF",
        "upload": "➕ Upload PDF (optional)",
        "pdf_override": "📘 Answer based on your uploaded PDF.",
        "info": "ℹ️ This system is for guidance only."
    },

    "Hinglish": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Ye chatbot WRD data aur aapke uploaded PDF se answer deta hai.",
        "query": "✍️ Apna sawaal likhiye",
        "button": "✅ Answer Pao",
        "search": "🔎 Documents dhoonde ja rahe hain...",
        "thinking": "🤖 Answer ban raha hai...",
        "answer": "🤖 Chatbot ka Answer:",
        "pdf": "📄 Use huye WRD PDF:",
        "download": "⬇️ PDF Download",
        "upload": "➕ Apna PDF Upload karein",
        "pdf_override": "📘 Answer sirf uploaded PDF se banaya gaya hai.",
        "info": "ℹ️ Ye system sirf guidance ke liye hai."
    }
}


# ---------------------------------------------------------
# 📚 WRD Knowledge Base Loader
# ---------------------------------------------------------

@st.cache_resource
def load_kb_and_vectorizer():
    if not os.path.exists("wrd_kb.json"):
        st.error("❌ wrd_kb.json missing!")
        st.stop()

    with open("wrd_kb.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    texts, meta = [], []

    for d in docs:
        texts.append(f"{d.get('title', '')}\n\n{d.get('text', '')}")
        meta.append({
            "title": d.get("title", ""),
            "url": d.get("url", ""),
            "type": d.get("type", ""),
        })

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    return docs, meta, vectorizer, matrix


def retrieve_context(query, vectorizer, matrix, docs, meta, top_k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idxs = sims.argsort()[::-1][:top_k]

    chunks = []
    pdf_sources = []

    for i in idxs:
        chunks.append(docs[i]["text"][:900])
        if meta[i]["type"].lower() == "pdf":
            pdf_sources.append(meta[i])

    return "\n\n----\n\n".join(chunks), pdf_sources


# ---------------------------------------------------------
# 📄 PDF Reader
# ---------------------------------------------------------

def read_uploaded_pdf(uploaded):
    if pdfplumber is None:
        return "❌ PDF reader not supported."

    text = ""
    with pdfplumber.open(uploaded) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text[:5000]


# ---------------------------------------------------------
# 🤖 GROQ CLOUD LLM (WITH CHAT HISTORY)
# ---------------------------------------------------------

def ask_llm_cloud(query, context, history, selected_lang):

    if "GROQ_API_KEY" not in st.secrets:
        return "❌ GROQ_API_KEY missing in Streamlit Secrets!"

    key = st.secrets["GROQ_API_KEY"]

    prompt = f"""
You are an official WRD assistant.
Answer in this language: {selected_lang}
Use BOTH chat history and the WRD context.
Give a long, detailed, step-by-step answer.

Chat History:
{history}

Context:
{context}

User Question:
{query}
"""

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a helpful WRD assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2
    }

    res = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    data = res.json()

    if "choices" not in data:
        return f"❌ Groq Error: {data}"

    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------
# 🎨 UI (No Changes — Same As Your Original)
# ---------------------------------------------------------

st.set_page_config(page_title="WRD AI Chatbot", layout="centered")
init_chat_history()

selected_lang = st.selectbox("🌐 Select Language / भाषा चुनें", list(LANGUAGES.keys()))
ui = LANGUAGES[selected_lang]

st.title(ui["title"])
st.markdown(ui["desc"])

uploaded_pdf = st.file_uploader(ui["upload"], type=["pdf"])

docs, meta, vectorizer, matrix = load_kb_and_vectorizer()

query = st.text_area(ui["query"], height=140)
top_k = st.slider("📄 Top Documents", 1, 5, 3)

pdf_sources = []  # prevent undefined error

if st.button(ui["button"]):

    history = get_history_for_llm()

    if uploaded_pdf:
        context = read_uploaded_pdf(uploaded_pdf)
        pdf_sources = []
        st.info(ui["pdf_override"])

    else:
        context, pdf_sources = retrieve_context(
            query, vectorizer, matrix, docs, meta, top_k
        )

    with st.spinner(ui["thinking"]):
        answer = ask_llm_cloud(query, context, history, selected_lang)

    add_message("user", query)
    add_message("assistant", answer)

# ---------------------------------------------------------
# 💬 Show Chat History
# ---------------------------------------------------------

st.subheader(ui["answer"])

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**🧑 User:** {msg['message']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['message']}")

# ---------------------------------------------------------
# PDF Info
# ---------------------------------------------------------

if pdf_sources:
    st.subheader(ui["pdf"])
    for p in pdf_sources:
        st.markdown(f"📄 **{p['title']}**")
        st.markdown(f"[{ui['download']}]({p['url']})")

# ---------------------------------------------------------
# Clear Chat Button
# ---------------------------------------------------------

if st.button("🗑 Clear Chat"):
    clear_history()
    st.success("Chat cleared!")

st.info(ui["info"])
