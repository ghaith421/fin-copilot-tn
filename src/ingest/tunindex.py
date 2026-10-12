import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import time
import json
from config import DATA_RAW

# Symboles possibles pour le TUNINDEX sur ilBoursa
TUNINDEX_SYMBOLS = ["TUNINDEX", "TUN", "TUNINDEX_INDEX"]

LENGTH = 1000
EPOCH = pd.Timestamp("1970-01-01")


def fetch_tunindex():
    """Recupere l'historique du TUNINDEX via ilBoursa."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[tunindex] Playwright non installe")
        return pd.DataFrame()

    resultats = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            locale="fr-FR",
        )
        page = context.new_page()

        print("[tunindex] Chargement de la page ilBoursa...")
        try:
            page.goto("https://www.ilboursa.com/marches/aaz",
                      timeout=60000, wait_until="domcontentloaded")
            time.sleep(4)
        except Exception as ex:
            print(f"[tunindex] Erreur : {ex}")

        for symbol in TUNINDEX_SYMBOLS:
            url = (f"https://www.ilboursa.com/api/charting/GetTicksEOD"
                   f"?symbol={symbol}&length={LENGTH}&period=0&addIntradayArray=false")
            try:
                response = context.request.get(
                    url,
                    headers={
                        "Referer": "https://www.ilboursa.com/marches/aaz",
                        "Accept": "application/json, text/plain, */*",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                )
                if response.status != 200:
                    print(f"[tunindex] {symbol} : HTTP {response.status}")
                    continue

                data = response.json()
                points = []
                if isinstance(data, dict):
                    points = data.get("QuoteTab", [])
                elif isinstance(data, list):
                    points = data

                if len(points) > 0:
                    print(f"[tunindex] {symbol} : {len(points)} points OK")
                    resultats[symbol] = points
                    break
                else:
                    print(f"[tunindex] {symbol} : vide")
            except Exception as ex:
                print(f"[tunindex] {symbol} : {str(ex)[:80]}")
            time.sleep(0.5)

        browser.close()

    if not resultats:
        print("[tunindex] Aucun symbole trouve")
        return pd.DataFrame()

    symbol = list(resultats.keys())[0]
    points = resultats[symbol]

    rows = []
    for pt in points:
        try:
            date = EPOCH + pd.Timedelta(days=int(pt.get("d", 0)))
        except Exception:
            continue
        rows.append({
            "date": date,
            "ticker": "TUNINDEX",
            "open": pt.get("o"),
            "high": pt.get("h"),
            "low": pt.get("l"),
            "close": pt.get("c"),
            "volume": pt.get("v"),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)

    out = f"{DATA_RAW}/tunindex.parquet"
    df.to_parquet(out)
    print(f"[tunindex] {len(df)} lignes sauvegardees -> {out}")
    print(f"[tunindex] Periode : {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


if __name__ == "__main__":
    fetch_tunindex()