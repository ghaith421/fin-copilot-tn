import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import time
from config import DATA_RAW

# Mapping ticker BVMT -> nom exact sur ilBoursa
BANK_NAME_MAP = {
    "BIAT":  "BIAT",
    "ATB":   "ARAB TUNISIAN BANK",
    "AB":    "ATTIJARI BANK",
    "AMEN":  "AMEN BANK",
    "UIB":   "UIB",
    "BNA":   "BANQUE NATIONALE AGRICOLE",
    "BH":    "BH BANK",
    "STB":   "STB BANK",
    "BT":    "BANQUE DE TUNISIE",
    "BTE":   "BTE (ADP)",
    "UBCI":  "UBCI",
    "WIFAK": "WIFAK INT BANK",
}

# Pour matcher dans le tableau ilBoursa
ILBOURSA_NAMES = set(BANK_NAME_MAP.values())

ILBOURSA_URL = "https://www.ilboursa.com/marches/aaz"


def scrape_ilboursa():
    """Scrape ilBoursa via Playwright (Edge)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[bvmt] Playwright non installe")
        return pd.DataFrame()

    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="fr-FR",
            viewport={"width": 1400, "height": 900},
        )
        page = context.new_page()

        try:
            print(f"[bvmt] Chargement de {ILBOURSA_URL}...")
            page.goto(ILBOURSA_URL, timeout=60000, wait_until="domcontentloaded")
            time.sleep(5)

            for _ in range(3):
                page.evaluate("window.scrollBy(0, 1000)")
                time.sleep(1)

            table_data = page.evaluate("""() => {
                const rows = [];
                document.querySelectorAll('table tr').forEach(tr => {
                    const cells = Array.from(tr.querySelectorAll('td, th'))
                        .map(c => c.textContent.trim());
                    if (cells.length >= 2) rows.push(cells);
                });
                return rows;
            }""")

            print(f"[bvmt] {len(table_data)} lignes recuperees")

            for cells in table_data:
                # Le nom est dans cells[0]
                nom = cells[0].strip().upper()
                if nom in ILBOURSA_NAMES:
                    # Le prix de cloture est dans cells[6] (Dernier)
                    # cells: Nom, Ouverture, +Haut, +Bas, Volume titres, Volume DT, Dernier, Variation
                    try:
                        close = float(cells[6].replace(",", ".").replace(" ", "").replace("\xa0", ""))
                    except Exception:
                        close = None

                    if close and close > 0:
                        # Retrouver le ticker
                        ticker = next(k for k, v in BANK_NAME_MAP.items() if v == nom)
                        rows.append({
                            "ticker": ticker,
                            "close": close,
                            "source": "ilboursa",
                        })

        except Exception as ex:
            print(f"[bvmt] Erreur : {ex}")
        finally:
            browser.close()

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="first")
        return df
    return pd.DataFrame()


def update_prices_history(new_df):
    if len(new_df) == 0:
        return pd.DataFrame()

    today = pd.Timestamp.now().normalize()
    new_df = new_df.copy()
    new_df["date"] = today

    path = f"{DATA_RAW}/bvmt_prices.parquet"

    if os.path.exists(path):
        old = pd.read_parquet(path)
        old["date"] = pd.to_datetime(old["date"])
        combined = pd.concat([old, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined.to_parquet(path)
    print(f"[bvmt] Historique : {len(combined)} lignes -> {path}")
    return combined


def ingest_bvmt_prices():
    print("[bvmt] Scraping automatique via Edge...")
    df = scrape_ilboursa()

    if len(df) == 0:
        print("[bvmt] Aucun cours recupere")
        return pd.DataFrame()

    print(f"[bvmt] {len(df)} cours recuperes")
    print(df[["ticker", "close"]])

    combined = update_prices_history(df)
    return combined


if __name__ == "__main__":
    ingest_bvmt_prices()