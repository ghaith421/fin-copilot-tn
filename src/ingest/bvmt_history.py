import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import time
import json
from config import DATA_RAW

# Symboles ilBoursa -> tickers BVMT
SYMBOL_MAP = {
    "BIAT":  "BIAT",
    "ATB":   "ATB",
    "TJARI": "AB",     # Attijari Bank
    "AB":    "AMEN",   # Amen Bank
    "UIB":   "UIB",
    "BNA":   "BNA",
    "BH":    "BH",
    "STB":   "STB",
    "BT":    "BT",
    "BTE":   "BTE",
    "UBCI":  "UBCI",
    "WIFAK": "WIFAK",
}

LENGTH = 1000

# Epoch correct : 1970-01-01 (Unix epoch)
EPOCH = pd.Timestamp("1970-01-01")


def scrape_history():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[hist] Playwright non installe")
        return {}

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

        print("[hist] Chargement de la page ilBoursa...")
        try:
            page.goto("https://www.ilboursa.com/marches/biat",
                      timeout=60000, wait_until="domcontentloaded")
            time.sleep(4)
        except Exception as ex:
            print(f"[hist] Erreur chargement : {ex}")

        for symbol, ticker in SYMBOL_MAP.items():
            url = (f"https://www.ilboursa.com/api/charting/GetTicksEOD"
                   f"?symbol={symbol}&length={LENGTH}&period=0&addIntradayArray=false")

            try:
                response = context.request.get(
                    url,
                    headers={
                        "Referer": "https://www.ilboursa.com/marches/biat",
                        "Accept": "application/json, text/plain, */*",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                )

                if response.status != 200:
                    print(f"[hist] {ticker} ({symbol}) : HTTP {response.status}")
                    continue

                data = response.json()

                points = []
                if isinstance(data, dict):
                    points = data.get("QuoteTab", [])
                elif isinstance(data, list):
                    points = data

                if len(points) > 0:
                    print(f"[hist] {ticker} ({symbol}) : {len(points)} points OK")
                    resultats[ticker] = {"symbol": symbol, "points": points}
                else:
                    print(f"[hist] {ticker} ({symbol}) : vide")

            except Exception as ex:
                print(f"[hist] {ticker} ({symbol}) : {str(ex)[:80]}")

            time.sleep(0.5)

        browser.close()

    return resultats


def parse_history(points, ticker):
    """Parse les points JSON en DataFrame."""
    rows = []
    for point in points:
        try:
            # Epoch = 1970-01-01 (Unix)
            date = EPOCH + pd.Timedelta(days=int(point.get("d", 0)))
        except Exception:
            continue
        rows.append({
            "date": date,
            "ticker": ticker,
            "open": point.get("o"),
            "high": point.get("h"),
            "low": point.get("l"),
            "close": point.get("c"),
            "volume": point.get("v"),
        })
    return pd.DataFrame(rows)


def ingest_history():
    print(f"[hist] Recuperation de l'historique ({LENGTH} jours)...")
    resultats = scrape_history()

    if not resultats:
        print("[hist] Aucun historique recupere")
        return pd.DataFrame()

    print(f"\n[hist] {len(resultats)} banques recuperees")

    all_dfs = []
    for ticker, info in resultats.items():
        df = parse_history(info["points"], ticker)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("[hist] Aucune donnee parseable")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[["date", "ticker", "open", "high", "low", "close", "volume"]]
    combined = combined.dropna(subset=["close"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = f"{DATA_RAW}/bvmt_history.parquet"
    combined.to_parquet(out)
    print(f"\n[hist] {len(combined)} lignes sauvegardees -> {out}")
    print(f"[hist] Periode : {combined['date'].min().date()} -> {combined['date'].max().date()}")
    print("\n[hist] Points par banque :")
    print(combined.groupby("ticker").size())

    return combined


if __name__ == "__main__":
    ingest_history()