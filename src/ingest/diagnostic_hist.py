import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import json


def diagnostic():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="msedge")
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            locale="fr-FR",
        )
        page = context.new_page()

        # Capturer les requetes reseau
        requests_captured = []

        def on_request(request):
            if "GetTicksEOD" in request.url:
                requests_captured.append({
                    "url": request.url,
                    "headers": dict(request.headers),
                })

        def on_response(response):
            if "GetTicksEOD" in response.url:
                try:
                    body = response.text()
                except Exception as ex:
                    body = f"(erreur: {ex})"
                requests_captured.append({
                    "response_url": response.url,
                    "status": response.status,
                    "body_preview": body[:500],
                })

        page.on("request", on_request)
        page.on("response", on_response)

        print("=== Chargement de la page BIAT ===")
        page.goto("https://www.ilboursa.com/marches/biat",
                  timeout=60000, wait_until="domcontentloaded")

        print("Attente 10 sec pour laisser les requetes partir...")
        time.sleep(10)

        print(f"\n=== {len(requests_captured)} evenements captures ===\n")
        for i, evt in enumerate(requests_captured):
            print(f"--- Evenement {i+1} ---")
            for k, v in evt.items():
                if k == "headers":
                    print(f"  {k}:")
                    for hk, hv in v.items():
                        if "cookie" in hk.lower() or "auth" in hk.lower():
                            print(f"    {hk}: {hv[:80]}...")
                        else:
                            print(f"    {hk}: {hv}")
                elif k == "body_preview":
                    print(f"  {k}: {v}")
                else:
                    print(f"  {k}: {v}")
            print()

        # Test manuel de l'API
        print("\n=== TEST MANUEL API ===")
        test_url = "https://www.ilboursa.com/api/charting/GetTicksEOD?symbol=BIAT&length=180&period=0&addIntradayArray=false"
        try:
            response = context.request.get(
                test_url,
                headers={
                    "Referer": "https://www.ilboursa.com/marches/biat",
                    "Accept": "application/json, text/plain, */*",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            print(f"Status: {response.status}")
            print(f"Headers: {dict(response.headers)}")
            body = response.text()
            print(f"Body (500 premiers car): {body[:500]}")
        except Exception as ex:
            print(f"Erreur: {ex}")

        browser.close()


if __name__ == "__main__":
    diagnostic()