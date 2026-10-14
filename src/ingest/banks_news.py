import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import feedparser
import requests
import pandas as pd
import time
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from config import DATA_RAW

# =====================================================
# BANQUES (tickers BVMT)
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

# =====================================================
# MOTS-CLES SECTORIELS
# =====================================================
SECTOR_KEYWORDS = [
    "bancaire", "banque", "banques",
    "bct", "banque centrale",
    "credit", "crédit", "credits", "crédits",
    "taux directeur",
    "pnb", "produit net bancaire",
    "depot", "dépôt", "depots", "dépôts",
    "bvmt", "tunindex", "bourse de tunis",
    "actionnaire", "dividende", "resultat", "résultat",
    "bilan", "capital", "titre", "bourse",
]

# =====================================================
# RSS DIRECTS
# =====================================================
RSS_DIRECT = {
    "ilboursa": [
        "https://www.ilboursa.com/rss/actualites",
        "https://www.ilboursa.com/rss",
    ],
    "tustex": [
        "https://www.tustex.com/rss.xml",
        "https://www.tustex.com/fr/rss.xml",
    ],
    "businessnews": [
        "https://www.businessnews.com.tn/rss.xml",
        "https://www.businessnews.com.tn/rss",
    ],
    "lapresse": [
        "https://www.lapresse.tn/feed/",
        "https://www.lapresse.tn/rss.xml",
    ],
    "africanmanager": [
        "https://www.africanmanager.com/feed/",
        "https://www.africanmanager.com/rss.xml",
    ],
    "webmanagercenter": [
        "https://www.webmanagercenter.com/feed/",
        "https://www.webmanagercenter.com/rss.xml",
    ],
    "leaders": [
        "https://www.leaders.com.tn/rss.xml",
        "https://www.leaders.com.tn/feed",
    ],
    "kapitalis": [
        "https://kapitalis.com/tunisie/feed/",
    ],
    "tekiano": [
        "https://www.tekiano.com/feed/",
    ],
    "tunisienumerique": [
        "https://www.tunisienumerique.com/feed/",
    ],
}

DAYS_BACK = 60


def google_news_url(query):
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=fr&gl=TN&ceid=TN:fr"


def _clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def fetch_google_news(ticker, keywords, max_per_query=50):
    rows = []
    seen_titles = set()

    for kw in keywords[:2]:
        url = google_news_url(f"{kw} tunisie")
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:max_per_query]:
                title = _clean_text(getattr(e, "title", ""))
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
                    "summary": _clean_text(getattr(e, "summary", "")),
                    "source": source,
                    "url": getattr(e, "link", ""),
                    "tickers": None,
                })
        except Exception as ex:
            print(f"[banks] Google News erreur {ticker}/{kw}: {ex}")
        time.sleep(0.5)

    return rows


def fetch_direct_rss():
    all_rows = []
    successful_sources = []

    for source_name, urls in RSS_DIRECT.items():
        source_ok = False
        for url in urls:
            try:
                feed = feedparser.parse(url)
                if not feed.entries:
                    continue
                count = 0
                for e in feed.entries:
                    title = _clean_text(getattr(e, "title", ""))
                    if not title or len(title) < 15:
                        continue

                    try:
                        date = datetime(*e.published_parsed[:6]) if hasattr(e, "published_parsed") and e.published_parsed else datetime.utcnow()
                    except Exception:
                        date = datetime.utcnow()

                    all_rows.append({
                        "date": date,
                        "title": title,
                        "summary": _clean_text(getattr(e, "summary", "")),
                        "source": source_name,
                        "url": getattr(e, "link", ""),
                        "tickers": None,
                    })
                    count += 1

                if count > 0:
                    successful_sources.append(f"{source_name} ({count})")
                    source_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.3)

    if successful_sources:
        print(f"[banks] RSS directs OK : {', '.join(successful_sources)}")

    return all_rows


# =====================================================
# MATCHING STRICT
# =====================================================
def _match_word_exact(text, word):
    """Verifie si un mot apparait en tant que MOT ENTIER."""
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def detect_bank_strict(text):
    """
    Detecte la banque avec correspondance STRICTE.
    - Tickers courts (<=5 char) : mot entier uniquement
    - Noms longs : substring OK
    """
    text_lower = text.lower()

    for ticker, keywords in BANKS.items():
        for kw in keywords:
            kw_lower = kw.lower()
            if len(kw) <= 5:
                if _match_word_exact(text, kw):
                    return ticker
            else:
                if kw_lower in text_lower:
                    return ticker
    return None


def is_sector_news(text):
    """Verifie si la news est bien sectorielle."""
    text_lower = (text or "").lower()
    return any(_match_word_exact(text, kw) for kw in SECTOR_KEYWORDS)


def tag_news_strict(df):
    """
    Tag STRICT (assoupli) :
    - Si ticker detecte (mot entier) -> assigne au ticker
    - Sinon si secteur bancaire -> SECTEUR
    - Sinon -> ignore
    """
    def match(row):
        text = (row["title"] or "") + ". " + (row.get("summary", "") or "")

        ticker = detect_bank_strict(text)
        sector = is_sector_news(text)

        # Ticker detecte -> on garde (meme sans mot sectoriel)
        if ticker:
            return ticker

        # Sinon secteur -> SECTEUR
        if sector:
            return "SECTEUR"

        return None

    df["tickers"] = df.apply(match, axis=1)
    df = df.dropna(subset=["tickers"])
    return df


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

    print("[banks] RSS directs tunisiens...")
    direct_rows = fetch_direct_rss()
    print(f"[banks]   {len(direct_rows)} articles des RSS directs")
    all_rows.extend(direct_rows)

    if not all_rows:
        print("[banks] Aucune news recuperee")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["title"]).dropna(subset=["title"])
    df = df[df["title"].str.len() > 15]

    # Filtre 60 jours
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    cutoff = pd.Timestamp.now() - timedelta(days=DAYS_BACK)
    n_before = len(df)
    df = df[df["date"] >= cutoff]
    n_after = len(df)
    print(f"[banks] Filtre {DAYS_BACK} jours : {n_before} -> {n_after} news")

    # Tag STRICT
    n_before_tag = len(df)
    df = tag_news_strict(df)
    n_after_tag = len(df)
    print(f"[banks] Filtre sectoriel : {n_before_tag} -> {n_after_tag} news")

    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    out = f"{DATA_RAW}/banks_news.parquet"
    df.to_parquet(out)
    print(f"\n[banks] TOTAL {len(df)} news -> {out}")
    print(df["tickers"].value_counts())
    return df


if __name__ == "__main__":
    ingest_banks_news()