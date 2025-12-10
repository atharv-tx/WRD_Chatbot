# 💧 WRD Chhattisgarh AI Chatbot  
_A Real-Data RAG + PDF Upload + Multi-Language AI Assistant for Water Resources Department_

---

## 📌 Project Overview

This project is an **AI-powered chatbot** built using **Streamlit + Ollama + RAG (Retrieval Augmented Generation)** that answers user queries using:

✅ Official **WRD Chhattisgarh website & PDF documents**  
✅ **User uploaded PDF files** (strict override mode)  
✅ **Multi-language UI: Hindi, English & Hinglish**

It is designed for:
- **Government Internships**
- **Citizen Information Systems**
- **Smart Governance Solutions**
- **AI Portfolio Projects**

---

## 🚀 Key Features

✅ Real data based chatbot (Not dummy data)  
✅ RAG using WRD website + PDFs  
✅ Strict **PDF Upload Override Mode**  
✅ Multi-language User Interface  
✅ Auto-continue for long, detailed answers  
✅ High-quality, step-by-step informational output  
✅ Only relevant PDFs shown with Download Button  
✅ Runs completely **offline** using **Ollama**  
✅ Safe `.gitignore` & production ready structure  

---

## 🧠 Technology Stack

| Purpose | Technology |
|--------|------------|
| Frontend | Streamlit |
| Backend | Python |
| AI Model | LLaMA 3.1 (via Ollama) |
| RAG Search | TF-IDF + Cosine Similarity |
| PDF Parsing | pdfplumber |
| Web Scraping | BeautifulSoup |
| Local LLM | Ollama |
| Data Storage | JSON Knowledge Base |

---

## 📁 Project Structure

WRD_Chatbot/
│
├── app.py # Main Streamlit App
├── fetch_wrd_data.py # WRD Website + PDF Scraper
├── wrd_kb.json # Knowledge Base
├── requirements.txt
├── README.md
├── .gitignore
└── venv/ (ignored)

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YourUsername/wrd-chatbot.git
cd wrd-chatbot
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Install & Run Ollama
👉 Download from: https://ollama.com

bash
Copy code
ollama pull llama3.1
ollama serve
5️⃣ Fetch WRD Data
bash
Copy code
python fetch_wrd_data.py
✅ This will create:

pgsql
Copy code
wrd_kb.json
6️⃣ Run the Chatbot
bash
Copy code
streamlit run app.py
📄 PDF Upload Mode
If the user uploads a PDF:

✅ The chatbot only reads that PDF

❌ WRD knowledge base is fully ignored

✅ Answers strictly come from the uploaded document

🌐 Multi-Language Support
You can switch the UI and answer language between:

✅ Hindi
✅ English
✅ Hinglish

🔐 Security & Best Practices
✅ .gitignore prevents:

Virtual environment upload

Cache files

Secret .env files

Log files

✅ No cloud API keys required
✅ Fully offline & secure

🧪 Example Use Cases
"Water Allotment की पूरी प्रक्रिया बताइए"

"Canal irrigation system ka structure samjhaiye"

"Upload PDF and ask from circular notification"

🎓 Internship & Academic Use
This project is suitable for:

Final year B.Tech projects

Smart India Hackathon ideas

Government department digitization

AI research & demos

👨‍💻 Developer
Atharv Singh Patle
B.Tech Student (Data Science & Generative AI)
Intern, Water Resources Department, Raipur
GitHub: https://github.com/YourUsername

⚠️ Disclaimer
This chatbot is for educational & internship demo purposes only.


#### LINK --- https://wrdchatbot-3ynp7jhdw4ngmjunsxh9r6.streamlit.app/
For official government decisions, always consult the official department.
