import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import pdfplumber
import io
import re

BASE = "https://www.cgwrd.in"

SEED_URLS = [
    BASE + "/",                              
    BASE + "/water-allotment-system/",
    BASE + "/water-rates/",
    BASE + "/organisation/functions",
    BASE + "/documents",
    BASE + "/acts-rules",
    BASE + "/circulars",
]

HEADERS = {"User-Agent": "WRD-Chatbot-Scraper/1.0"}


# -----------------------------------------------------------
# 🚀 CLEANING HELPERS
# -----------------------------------------------------------

def remove_repeating_blocks(text: str) -> str:
    """Remove long repeating lines / garbage blocks (like Local Bodies spam)"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Remove lines repeated more than 5 times
    clean = []
    seen = {}

    for l in lines:
        seen[l] = seen.get(l, 0) + 1
        if seen[l] <= 3:  # allow only up to 3 repetitions
            clean.append(l)

    return "\n".join(clean)


def is_garbage_text(text: str) -> bool:
    """Identify corrupted PDF text"""
    if len(text) < 80:
        return True

    # Too much repetition
    words = text.split()
    if len(words) > 0:
        repetition = len(words) - len(set(words))
        if repetition / len(words) > 0.35:  # > 35% repeated = garbage
            return True

    # Contains table-like junk
    if re.search(r"(-\s*){10,}", text):
        return True

    return False


def extract_pdf_text(url):
    print("[PDF] Extracting:", url)
    resp = requests.get(url, headers=HEADERS, timeout=40)
    resp.raise_for_status()

    with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
        pages = []
        for p in pdf.pages:
            t = p.extract_text()
            if not t:
                continue

            t = remove_repeating_blocks(t)

            if not is_garbage_text(t):
                pages.append(t)

        return "\n".join(pages)


def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove irrelevant navigation elements
    for tag in soup(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 40]  # meaningful only

    return "\n".join(lines)


# -----------------------------------------------------------
# 🚀 MAIN SCRAPER
# -----------------------------------------------------------

def build_kb():
    docs = []
    visited = set()

    for url in SEED_URLS:
        try:
            print("[HTML] Fetching:", url)
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            html = resp.text

            soup = BeautifulSoup(html, "html.parser")
            title = soup.title.get_text(strip=True) if soup.title else url

            cleaned = clean_html(html)
            docs.append({
                "url": url,
                "title": title,
                "text": cleaned,
                "type": "html"
            })
            visited.add(url)

            # Find PDFs inside page
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.lower().endswith(".pdf"):
                    continue

                pdf_url = urljoin(BASE, href)

                if pdf_url in visited:
                    continue

                try:
                    pdf_text = extract_pdf_text(pdf_url)
                    if len(pdf_text) < 200:
                        continue  # skip junk pdf

                    docs.append({
                        "url": pdf_url,
                        "title": a.get_text(strip=True) or "WRD PDF",
                        "text": pdf_text,
                        "type": "pdf",
                    })

                    visited.add(pdf_url)

                except Exception as e:
                    print("[WARN] PDF failed:", pdf_url, "->", e)

        except Exception as e:
            print("[WARN] Failed:", url, e)

    # Save KB
    with open("wrd_kb.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print("\n✅ CLEAN RAG DATASET GENERATED SUCCESSFULLY")
    print("📁 Saved as wrd_kb.json")
    print("📄 Total Documents:", len(docs))


if __name__ == "__main__":
    build_kb()
