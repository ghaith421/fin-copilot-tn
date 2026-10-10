import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
from playwright.sync_api import sync_playwright


def dump_ilboursa():
    """Affiche TOUT ce qu'ilBoursa renvoie dans son tableau."""
    print("=" * 60)
    print("=== ILBOURSA - CONTENU COMPLET ===")
    print("=" * 60)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="fr-FR",
        )
        page = context.new_page()

        try:
            page.goto("https://www.ilboursa.com/marches/aaz", timeout=60000, wait_until="domcontentloaded")
            time.sleep(5)
            for _ in range(3):
                page.evaluate("window.scrollBy(0, 1000)")
                time.sleep(1)

            data = page.evaluate("""() => {
                const rows = [];
                document.querySelectorAll('table tr').forEach(tr => {
                    const cells = Array.from(tr.querySelectorAll('td, th'))
                        .map(c => c.textContent.trim());
                    if (cells.length >= 2) rows.push(cells);
                });
                return rows;
            }""")

            print(f"\n{len(data)} lignes recuperees.\n")
            for i, row in enumerate(data):
                print(f"Ligne {i+1:3d} : {row}")
        except Exception as ex:
            print(f"Erreur : {ex}")
        finally:
            browser.close()


def check_investing():
    """Teste investing.com pour les banques tunisiennes."""
    print("\n" + "=" * 60)
    print("=== INVESTING.COM - TEST ===")
    print("=" * 60)

    # URLs candidates sur investing.com
    urls = {
        "BIAT":  "https://www.investing.com/equities/banque-internationale-arabe-de-tunisie",
        "ATB":   "https://www.investing.com/equities/arab-tunisian-bank",
        "AB":    "https://www.investing.com/equities/attijari-bank",
        "AMEN":  "https://www.investing.com/equities/amen-bank",
        "UIB":   "https://www.investing.com/equities/union-internationale-de-banques",
        "BNA":   "https://www.investing.com/equities/banque-nationale-agricole",
        "BH":    "https://www.investing.com/equities/banque-de-l-habitat",
        "STB":   "https://www.investing.com/equities/societe-tunisienne-de-banque",
        "BT":    "https://www.investing.com/equities/banque-de-tunisie",
        "BTE":   "https://www.investing.com/equities/banque-de-tunisie-et-des-emirats",
        "ZB":    "https://www.investing.com/equities/banque-zitouna",
        "UBCI":  "https://www.investing.com/equities/union-bancaire-pour-le-commerce-et-l-industrie",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="fr-FR",
        )
        page = context.new_page()

        for ticker, url in urls.items():
            try:
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                time.sleep(2)

                # Chercher un prix
                price = page.evaluate("""() => {
                    const el = document.querySelector('[data-test="instrument-price-last"]');
                    if (el) return el.textContent.trim();
                    return null;
                }""")

                title = page.title()
                if price:
                    print(f"✅ {ticker} : {price}  ({title[:50]})")
                else:
                    print(f"❌ {ticker} : pas de prix  ({title[:50]})")
            except Exception as ex:
                print(f"❌ {ticker} : {str(ex)[:80]}")

        browser.close()


if __name__ == "__main__":
    dump_ilboursa()
    check_investing()