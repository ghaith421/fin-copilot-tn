import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import requests
from bs4 import BeautifulSoup
import re
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

# Pour chaque banque : liste de slugs a tester
CANDIDATS = {
    "BIAT":  ["biat", "biat_bi", "banque-internationale-arabe-de-tunisie"],
    "ATB":   ["atb", "atb_bi", "arab-tunisian-bank", "arab-tunisian"],
    "AB":    ["ab", "ab_bi", "attijari-bank", "attijaribank", "attijari"],
    "AMEN":  ["amen", "amen_bi", "amen-bank", "amenbank"],
    "UIB":   ["uib", "uib_bi", "union-internationale-de-banques"],
    "BNA":   ["bna", "bna_bi", "banque-nationale-agricole"],
    "BH":    ["bh", "bh_bi", "bh-bank", "banque-de-l-habitat", "bhl"],
    "STB":   ["stb", "stb_bi", "societe-tunisienne-de-banque"],
    "BT":    ["bt", "bt_bi", "banque-de-tunisie", "bt-bank"],
    "BTE":   ["bte", "bte_bi", "banque-de-tunisie-et-des-emirats"],
    "ZB":    ["zb", "zb_bi", "zitouna", "zitouna-bank", "banque-zitouna"],
    "UBCI":  ["ubci", "ubci_bi", "union-bancaire-pour-le-commerce"],
}

BASE_URLS = [
    "https://www.ilboursa.com/marches/{}",
    "https://www.ilboursa.com/societes/{}",
]


def test_url(url):
    """Retourne (status, prix_extrait) pour une URL."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if r.status_code != 200:
            return r.status_code, None, None

        soup = BeautifulSoup(r.text, "lxml")
        title = soup.title.get_text(strip=True) if soup.title else ""

        # Chercher un prix dans les tables
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                for c in tr.find_all(["td", "th"]):
                    txt = c.get_text(strip=True)
                    m = re.match(r"^(\d{1,4}[.,]\d{1,3})$", txt.replace(" ", ""))
                    if m:
                        try:
                            v = float(txt.replace(",", ".").replace(" ", ""))
                            if 0.1 < v < 10000:
                                return 200, v, title
                        except Exception:
                            pass

        # Chercher dans les spans de classe "cours" ou "price"
        for el in soup.find_all(["span", "div"]):
            cls = " ".join(el.get("class", []) or [])
            if "cours" in cls.lower() or "price" in cls.lower() or "last" in cls.lower():
                m = re.search(r"(\d{1,4}[.,]\d{1,3})", el.get_text(strip=True))
                if m:
                    try:
                        v = float(m.group(1).replace(",", "."))
                        if 0.1 < v < 10000:
                            return 200, v, title
                    except Exception:
                        pass

        return 200, None, title
    except Exception as ex:
        return None, None, str(ex)[:80]


def discover():
    resultats = {}
    for ticker, slugs in CANDIDATS.items():
        print(f"\n=== {ticker} ===")
        trouve = False
        for slug in slugs:
            for base in BASE_URLS:
                url = base.format(slug)
                status, price, title = test_url(url)
                marker = "✅" if (status == 200 and price) else ("⚠️" if status == 200 else "❌")
                print(f"  {marker} {url}")
                if title and status == 200:
                    print(f"      -> {title[:80]}")
                if price:
                    print(f"      -> PRIX : {price}")
                if status == 200 and price:
                    resultats[ticker] = {"slug": slug, "url": url, "price": price, "title": title}
                    trouve = True
                    break
                time.sleep(0.3)
            if trouve:
                break
        if not trouve:
            resultats[ticker] = None

    print("\n\n========================================")
    print("RESULTAT FINAL")
    print("========================================")
    for t, info in resultats.items():
        if info:
            print(f"✅ {t} -> slug='{info['slug']}' prix={info['price']}")
        else:
            print(f"❌ {t} -> AUCUN slug trouve")

    return resultats


if __name__ == "__main__":
    discover()