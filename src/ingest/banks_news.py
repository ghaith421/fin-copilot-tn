import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import feedparser
import requests
import pandas as pd
import time
from datetime import datetime
from bs4 import BeautifulSoup
from config import DATA_RAW

# =====================================================
# 12 BANQUES COTEES A LA BVMT
# =====================================================
BANKS = {
    "BIAT":  ["BIAT", "Banque Internationale Arabe de Tunisie"],
    "ATB":   ["ATB", "Arab Tunisian Bank"],
    "AB":    ["Attijari Bank", "Attijari bank Tunisie"],
    "AMEN":  ["Amen Bank", "Amen bank"],
    "UIB":   ["UIB", "Union Internationale de Banques"],
    "BNA":   ["BNA", "Banque Nationale Agricole"],
    "BH":    ["BH Bank", "Banque de l'Habitat"],
    "STB":   ["STB", "Societe Tunisienne de Banque"],
    "BT":    ["Banque de Tunisie"],
    "BTE":   ["BTE", "Banque de Tunisie et des Emirats"],
    "ZB":    ["Zitouna Bank", "Banque Zitouna"],
    "UBCI":  ["UBCI", "Union Bancaire pour le Commerce et l'Industrie"],
}

# Mots-cles du secteur bancaire tunisien
SECTOR_KEYWORDS = [
    "banque", "bancaire", "credit", "crédit",
    "bvmt", "bourse", "tunisie",
    "PNB", "produit net bancaire",
    "taux directeur", "BCT", "banque centrale",
    "dividende", "résultat", "resultat",
    "assemblée générale", "assemblee generale",
]


# =====================================================
# GOOGLE NEWS RSS (source fiable 100%)
# =====================================================
def google_news_url(query):
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=fr&gl=TN&ceid=TN:fr"


def fetch_google_news(ticker, keywords, max_per_query=30):
    rows = []
    seen_titles = set()

    for kw in keywords[:2]:
        url = google_news_url(f"{kw} tunisie")
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max_per_query]:
                title = getattr(e, "title", "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                try:
                    date = datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") and e.published_parsed else datetime.utcnow()
                except Exception:
                    date = datetime.utcnow()

                source = "google_news"
                if hasattr(e, "source") and e.source:
                    source = getattr(e.source, "title", "google_news")

                rows.append({
                    "date": date,
                    "title": title,
                    "summary": getattr(e, "summary", ""),
                    "source": source,
                    "url": getattr(e, "link", ""),
                    "tickers": ticker,
                })
        except Exception as ex:
            print(f"[banks] Google News erreur {ticker}/{kw}: {ex}")
        time.sleep(0.5)

    return rows


# =====================================================
# SCRAPING ILBOURSA + TUSTEX (complement)
# =====================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def scrape_generic(url, source_name):
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        candidates = soup.find_all("article") or soup.find_all(["h2", "h3"])
        seen = set()

        for a in candidates:
            link = a.find("a") if a.name != "a" else a
            if not link or not link.get("href"):
                continue
            title = link.get_text(strip=True)
            if len(title) < 20 or title in seen:
                continue

            ticker = detect_bank(title)
            if not ticker:
                continue

            seen.add(title)
            rows.append({
                "date": datetime.utcnow(),
                "title": title,
                "summary": title,
                "source": source_name,
                "url": link.get("href", ""),
                "tickers": ticker,
            })
    except Exception as ex:
        print(f"[banks] HTML erreur {url}: {ex}")
    return rows


def detect_bank(text):
    text = (text or "").lower()
    for ticker, keywords in BANKS.items():
        if any(kw.lower() in text for kw in keywords):
            return ticker
    return None


# =====================================================
# PIPELINE PRINCIPAL
# =====================================================
def ingest_banks_news():
    all_rows = []

    print("[banks] Google News par banque...")
    for ticker, keywords in BANKS.items():
        rows = fetch_google_news(ticker, keywords)
        print(f"[banks]   {ticker}: {len(rows)} articles")
        all_rows.extend(rows)

    print("[banks] Scraping ilBoursa + Tustex...")
    for url, name in [
        ("https://www.ilboursa.com/actualites", "ilboursa"),
        ("https://www.tustex.com/fr/actualites", "tustex"),
    ]:
        rows = scrape_generic(url, name)
        print(f"[banks]   {name}: {len(rows)} articles bancaires")
        all_rows.extend(rows)
        time.sleep(1)

    if not all_rows:
        print("[banks] Aucune news recuperee")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["title"]).dropna(subset=["title"])
    df = df[df["title"].str.len() > 15]
    df = df.dropna(subset=["tickers"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    out = f"{DATA_RAW}/banks_news.parquet"
    df.to_parquet(out)
    print(f"[banks] TOTAL {len(df)} news bancaires -> {out}")
    print(df["tickers"].value_counts())
    return df


if __name__ == "__main__":
    ingest_banks_news()