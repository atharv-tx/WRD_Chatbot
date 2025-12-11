# app.py
import streamlit as st
import json
import os
import requests
import io
import re
from typing import List, Tuple

# Optional PDF reader
try:
    import pdfplumber
except Exception:
    pdfplumber = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Configuration
# -------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"   # <-- recommended replacement for retired models

PDFS_FOLDER = "pdfs"          # local folder where you place WRD PDFs
KB_FILE = "wrd_kb.json"       # optional existing KB from your scraper

MAX_CONTEXT_CHARS = 3500      # guard to keep the prompt reasonably sized
HISTORY_LINES = 6             # how many recent chat lines to include for tone continuity
TOP_K_DOCS = 3                # default top-k retrieval

# -------------------------
# UI language strings
# -------------------------
LANGUAGES = {
    "हिंदी": {
        "title": "💧 जल संसाधन विभाग छत्तीसगढ़ – एआई चैटबॉट",
        "desc": "WRD दस्तावेज़ों और आपके अपलोड किए गए PDF के आधार पर उत्तर।",
        "query": "✍️ अपना सवाल लिखिए",
        "button": "✅ उत्तर प्राप्त करें",
        "thinking": "🤖 उत्तर तैयार किया जा रहा है...",
        "answer": "🤖 चैटबॉट का उत्तर:",
        "pdf": "📄 उपयोग किए गए PDF:",
        "download": "⬇️ PDF डाउनलोड करें",
        "upload": "➕ PDF अपलोड करें (वैकल्पिक)",
        "meta_builtin": "यह चैटबॉट WRD की वेबसाइट और PDF को वेब-स्क्रैप करके RAG के माध्यम से उत्तर देता है। अगर आप चैटबॉट के बारे में पूछते हैं, तो यह उत्तर सिस्टम-built विवरण देगा, PDF पर निर्भर नहीं होगा।",
        "no_api": "❌ GROQ_API_KEY Streamlit Secrets में नहीं मिला।",
        "pdf_not_supported": "⚠️ इस होस्ट पर PDF पढ़ने की सुविधा उपलब्ध नहीं है (pdfplumber missing).",
        "info": "ℹ️ यह प्रणाली मार्गदर्शन के लिए है; आधिकारिक जानकारी हेतु विभाग से संपर्क करें।"
    },
    "English": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Answers from WRD documents and your uploaded PDF.",
        "query": "✍️ Enter your question",
        "button": "✅ Get Answer",
        "thinking": "🤖 Generating answer...",
        "answer": "🤖 Chatbot Answer:",
        "pdf": "📄 Used PDFs:",
        "download": "⬇️ Download PDF",
        "upload": "➕ Upload PDF (optional)",
        "meta_builtin": "This chatbot collects WRD pages and PDFs and uses RAG to produce answers. Questions about the chatbot return a built-in explanation (not from PDFs).",
        "no_api": "❌ GROQ_API_KEY missing in Streamlit Secrets.",
        "pdf_not_supported": "⚠️ PDF reading is not available on this host (pdfplumber missing).",
        "info": "ℹ️ This system is for guidance only. For official decisions contact WRD."
    },
    "Hinglish": {
        "title": "💧 WRD Chhattisgarh – AI Chatbot",
        "desc": "Ye chatbot WRD documents aur uploaded PDF se jawab deta hai.",
        "query": "✍️ Apna sawaal likhiye",
        "button": "✅ Answer Pao",
        "thinking": "🤖 Answer ban raha hai...",
        "answer": "🤖 Chatbot Ka Answer:",
        "pdf": "📄 Use hue PDFs:",
        "download": "⬇️ PDF Download",
        "upload": "➕ PDF Upload (optional)",
        "meta_builtin": "Yeh chatbot WRD pages/PDFs ko scrape karke RAG se answer deta hai. Chatbot related questions me built-in explanation diya jayega (PDF par depend nahi karega).",
        "no_api": "❌ GROQ_API_KEY Streamlit Secrets me missing hai.",
        "pdf_not_supported": "⚠️ PDF read karne ka option is host par available nahi hai (pdfplumber missing).",
        "info": "ℹ️ Ye system sirf guidance ke liye hai."
    }
}

# -------------------------
# Chat history helpers
# -------------------------
def init_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def add_message(role: str, message: str):
    st.session_state.chat_history.append({"role": role, "message": message})

def clear_history():
    st.session_state.chat_history = []

def get_history_for_llm() -> str:
    """Return the last N chat lines for tone continuity. LLM should not treat them as facts."""
    history = st.session_state.get("chat_history", [])[-HISTORY_LINES*2:]
    out = ""
    for m in history:
        speaker = "User" if m["role"] == "user" else "Assistant"
        out += f"{speaker}: {m['message']}\n"
    return out

# -------------------------
# Meta-question detection
# -------------------------
META_PATTERNS = [
    r"\bwhat is this\b", r"\bwho are you\b", r"\bwhat can you do\b",
    r"chatbot", r"how does it work", r"kaise kaam", r"bot kya", r"your purpose",
    r"what is this chatbot", r"tell me about yourself"
]

def is_meta_question(q: str) -> bool:
    q_low = q.lower()
    # if includes domain-specific keywords, treat as normal question
    domain_keywords = ["water", "wrd", "irrigation", "dam", "canal", "ground water", "allotment", "permit", "permission"]
    if any(k in q_low for k in domain_keywords):
        return False
    return any(re.search(p, q_low) for p in META_PATTERNS)

# -------------------------
# Load KB (wrd_kb.json) and local pdfs folder
# -------------------------
@st.cache_resource
def load_kb_and_vectorizer(kb_file: str = KB_FILE, pdfs_folder: str = PDFS_FOLDER):
    docs = []
    meta = []

    # 1) Load existing KB JSON (if exists)
    if os.path.exists(kb_file):
        try:
            with open(kb_file, "r", encoding="utf-8") as f:
                kb_docs = json.load(f)
            for d in kb_docs:
                docs.append(d.get("text", "")[:10000])  # store truncated text
                meta.append({"title": d.get("title", ""), "url": d.get("url", ""), "type": d.get("type", "html")})
        except Exception as e:
            st.warning(f"Could not load {kb_file}: {e}")

    # 2) Load all local PDFs from pdfs/ folder (if exists)
    if os.path.isdir(pdfs_folder):
        if pdfplumber is None:
            st.warning("pdfplumber missing — local PDF folder will be listed but not read.")
        else:
            for fname in sorted(os.listdir(pdfs_folder)):
                if fname.lower().endswith(".pdf"):
                    path = os.path.join(pdfs_folder, fname)
                    try:
                        with pdfplumber.open(path) as pdf:
                            text_pages = []
                            for p in pdf.pages:
                                txt = p.extract_text()
                                if txt:
                                    text_pages.append(txt)
                            full_text = "\n".join(text_pages)
                        docs.append(full_text[:10000])
                        meta.append({"title": fname, "url": path, "type": "pdf"})
                    except Exception as e:
                        st.warning(f"Failed to read {fname}: {e}")

    if not docs:
        # avoid empty vectorizer crash
        docs = ["No documents found in KB or pdfs/ folder."]
        meta = [{"title": "No docs", "url": "", "type": "html"}]

    vectorizer = TfidfVectorizer().fit(docs)
    doc_matrix = vectorizer.transform(docs)
    return docs, meta, vectorizer, doc_matrix

# -------------------------
# Retrieval function
# -------------------------
def retrieve_context(query: str, docs: List[str], meta: List[dict], vectorizer: TfidfVectorizer, doc_matrix, top_k: int = TOP_K_DOCS) -> Tuple[str, List[dict]]:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    chunks = []
    used_pdfs = []
    for idx in top_idx:
        chunks.append(docs[idx][:MAX_CONTEXT_CHARS])
        if meta[idx].get("type", "").lower() == "pdf":
            used_pdfs.append(meta[idx])
    context = "\n\n----\n\n".join(chunks)
    return context, used_pdfs

# -------------------------
# ask Groq
# -------------------------
def ask_groq(query: str, context: str, history: str, lang_display: str) -> str:
    # check secrets
    if "GROQ_API_KEY" not in st.secrets:
        return st.session_state.ui_strings["no_api"]

    key = st.secrets["GROQ_API_KEY"]

    # Build role-aware prompt:
    # Rules: if context contains factual WRD content, LLM should use it (RAG).
    final_prompt = f"""
You are an expert assistant for the Water Resources Department (WRD) of Chhattisgarh.
Answer in the same language as the UI selection: {lang_display}.

RULES:
1) If the user's question is factual about WRD (water, allotment, permits, schemes, acts), PRIORITIZE the RAG context and cite the context when appropriate.
2) If context does NOT contain the answer, you may answer using general domain knowledge but clearly label anything not found in WRD documents as 'Suggested / General Guidance'.
3) Use chat history only for tone/continuity — do NOT treat chat history as authoritative facts for WRD procedures.
4) Give a long, step-by-step and precise answer (but avoid unnecessary repetition). If asked for short answer, provide a concise summary followed by details.

Chat History (tone only):
{history}

RAG Context (use this first, if available):
{context}

User Question:
{query}

Answer (start now):
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful, accurate WRD assistant."},
            {"role": "user", "content": final_prompt}
        ],
        # you can tune temperature/top_p here
        "temperature": 0.15,
        "max_tokens": 1400
    }

    try:
        resp = requests.post(GROQ_API_URL, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=90)
        if resp.status_code != 200:
            return f"❌ Groq API Error {resp.status_code}: {resp.text}"
        data = resp.json()
        # expected structure: choices[0].message.content
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ Groq returned unexpected response: {data}"
    except Exception as e:
        return f"❌ Network / Groq request failed: {e}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="WRD AI Chatbot (Groq 70B)", layout="centered")
init_chat_history()

# Language selection
selected_lang = st.selectbox("🌐 Select Language / भाषा चुनें", list(LANGUAGES.keys()))
ui = LANGUAGES[selected_lang]
st.session_state.ui_strings = ui  # make available to functions

st.title(ui["title"])
st.markdown(ui["desc"])

# File uploader (override)
uploaded_pdf = st.file_uploader(ui["upload"], type=["pdf"])

# Load KB + pdfs folder
with st.spinner("Loading knowledge base and PDFs..."):
    docs, meta, vectorizer, doc_matrix = load_kb_and_vectorizer()

# Inputs
query = st.text_area(ui["query"], height=140)
top_k = st.slider("📄 Documents to retrieve (top-k)", 1, 5, TOP_K_DOCS)

# Prepare pdf_sources variable for safe display outside button
pdf_sources_for_display = []

if st.button(ui["button"]):
    user_q = query.strip()
    if not user_q:
        st.warning("❗ Please enter a question.")
    else:
        # Meta-question?
        if is_meta_question(user_q):
            # built-in response
            answer = ui["meta_builtin"]
            add_message("user", user_q)
            add_message("assistant", answer)
            pdf_sources_for_display = []  # no PDF sources for meta answers
        else:
            # Normal RAG flow (or uploaded-PDF override)
            if uploaded_pdf:
                if pdfplumber is None:
                    context_text = ui["pdf_not_supported"]
                else:
                    try:
                        with pdfplumber.open(uploaded_pdf) as pdf:
                            pages = []
                            for p in pdf.pages:
                                txt = p.extract_text() or ""
                                pages.append(txt)
                        context_text = "\n".join(pages)[:MAX_CONTEXT_CHARS]
                    except Exception as e:
                        context_text = f"⚠️ Failed to read uploaded PDF: {e}"
                pdf_sources_for_display = [{"title": uploaded_pdf.name, "url": "uploaded_pdf", "type": "pdf"}]
            else:
                # retrieve from KB
                context_text, pdf_sources_for_display = retrieve_context(user_q, docs, meta, vectorizer, doc_matrix, top_k=top_k)

            # Build short history (tone only)
            history_for_tone = get_history_for_llm()

            with st.spinner(ui["thinking"]):
                answer = ask_groq(user_q, context_text, history_for_tone, selected_lang)

            # Save to history
            add_message("user", user_q)
            add_message("assistant", answer)

        # Display answer & sources
        st.subheader(ui["answer"])
        st.success(answer)

        # Show only used PDFs (if any) and provide download link or path
        if pdf_sources_for_display:
            st.subheader(ui["pdf"])
            for s in pdf_sources_for_display:
                title = s.get("title") or s.get("url")
                url = s.get("url", "")
                if url.startswith("http://") or url.startswith("https://"):
                    st.markdown(f"- 📄 **{title}** — [{ui['download']}]({url})")
                elif url == "uploaded_pdf":
                    # show a download button for the uploaded file
                    try:
                        b = uploaded_pdf.getvalue()
                        st.download_button(label=f"{ui['download']} — {title}", data=b, file_name=title, mime="application/pdf")
                    except Exception:
                        st.markdown(f"- 📄 **{title}** (uploaded)")
                else:
                    # local path (pdf in pdfs/ folder)
                    if os.path.exists(url):
                        with open(url, "rb") as f:
                            data = f.read()
                        st.download_button(label=f"{ui['download']} — {title}", data=data, file_name=title, mime="application/pdf")
                    else:
                        st.markdown(f"- 📄 **{title}** — (path: {url})")

# Show chat history (latest at bottom)
st.subheader(ui["answer"])
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 User:** {msg['message']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['message']}")
else:
    st.info("No conversation yet — ask a question to start.")

# Clear chat button
if st.button("🗑 Clear Conversation"):
    clear_history()
    st.success("Conversation cleared.")

st.info(ui["info"])
