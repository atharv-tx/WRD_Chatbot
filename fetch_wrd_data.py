import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import pdfplumber
import io

BASE = "https://www.cgwrd.in"

# ✅ WRD के कुछ important public pages (जरूरत अनुसार बढ़ा सकते हो)
SEED_URLS = [
    BASE + "/",                          
    BASE + "/water-allotment-system/",   
    BASE + "/water-rates/",              
    BASE + "/announcement",              
    BASE + "/organisation/functions",    
]

HEADERS = {
    "User-Agent": "WRD-Chatbot-Intern-Project/1.0 (Educational use only)"
}


def fetch_page(url: str) -> str:
    print(f"[INFO] Fetching HTML: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["nav", "footer", "script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 30]
    return "\n".join(lines)


# ✅ ✅ PDF TEXT EXTRACTOR
def extract_pdf_text(pdf_url: str) -> str:
    print(f"[INFO] Downloading PDF: {pdf_url}")
    resp = requests.get(pdf_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        pages = []
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)


def build_kb():
    docs = []
    visited_urls = set()

    # ✅ HTML pages fetch
    for url in SEED_URLS:
        try:
            html = fetch_page(url)
            text = clean_text(html)

            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url

            docs.append({
                "url": url,
                "title": title,
                "text": text,
                "type": "html"
            })

            visited_urls.add(url)

            # ✅ Page ke andar ke PDF links dhundhna
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = urljoin(BASE, href)
                    if pdf_url not in visited_urls:
                        try:
                            pdf_text = extract_pdf_text(pdf_url)
                            docs.append({
                                "url": pdf_url,
                                "title": a.get_text(strip=True) or "WRD PDF Document",
                                "text": pdf_text,
                                "type": "pdf"
                            })
                            visited_urls.add(pdf_url)
                        except Exception as e:
                            print(f"[WARN] PDF failed: {pdf_url} -> {e}")

        except Exception as e:
            print(f"[WARN] HTML page failed: {url} -> {e}")

    print(f"\n✅ TOTAL DOCUMENTS COLLECTED (HTML + PDF): {len(docs)}")

    with open("wrd_kb.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print("✅ Knowledge base saved as wrd_kb.json")


if __name__ == "__main__":
    build_kb()
