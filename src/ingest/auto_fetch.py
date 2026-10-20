import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from datetime import datetime
from config import DATA_RAW


def needs_refresh(path, max_age_minutes=30):
    """Verifie si un fichier doit etre rafraichi."""
    if not os.path.exists(path):
        return True
    age_min = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).total_seconds() / 60
    return age_min > max_age_minutes


def refresh_news():
    """Rafraichit les news bancaires si necessaire."""
    path = f"{DATA_RAW}/banks_news.parquet"
    if not needs_refresh(path, max_age_minutes=30):
        print(f"[auto] News deja fraiches")
        return False

    print("[auto] Rafraichissement des news...")
    try:
        from src.ingest.banks_news import ingest_banks_news
        ingest_banks_news()
        return True
    except Exception as ex:
        print(f"[auto] Erreur news : {ex}")
        return False


def refresh_prices():
    """Rafraichit les prix BVMT si necessaire."""
    path = f"{DATA_RAW}/bvmt_prices.parquet"
    if not needs_refresh(path, max_age_minutes=60):
        print(f"[auto] Prix deja frais")
        return False

    print("[auto] Rafraichissement des prix...")
    try:
        from src.ingest.bvmt_prices import ingest_bvmt_prices
        ingest_bvmt_prices()
        return True
    except Exception as ex:
        print(f"[auto] Erreur prix : {ex}")
        return False


def refresh_sentiment():
    """Rafraichit le sentiment avec le modele ML finance (avec fallback)."""
    path = f"{DATA_RAW}/banks_news_scored.parquet"
    news_path = f"{DATA_RAW}/banks_news.parquet"

    if os.path.exists(path) and os.path.exists(news_path):
        if os.path.getmtime(news_path) <= os.path.getmtime(path):
            print(f"[auto] Sentiment deja a jour")
            return False

    print("[auto] Analyse du sentiment (modele ML finance)...")
    try:
        from src.nlp.sentiment_ml import score_news_ml
        score_news_ml()
        return True
    except Exception as ex:
        print(f"[auto] Erreur ML : {ex}")
        print("[auto] Fallback sur mots-cles...")
        try:
            from src.nlp.sentiment_fr import score_news_keywords_only
            score_news_keywords_only()
            return True
        except Exception as ex2:
            print(f"[auto] Erreur fallback : {ex2}")
            return False


def refresh_history():
    """Rafraichit l'historique des prix BVMT (1 fois par jour)."""
    path = f"{DATA_RAW}/bvmt_history.parquet"
    if not needs_refresh(path, max_age_minutes=24 * 60):
        print(f"[auto] Historique deja frais")
        return False

    print("[auto] Rafraichissement de l'historique des prix...")
    try:
        from src.ingest.bvmt_history import ingest_history
        ingest_history()
        return True
    except Exception as ex:
        print(f"[auto] Erreur historique : {ex}")
        return False


def auto_update_all():
    """Lance toutes les mises a jour automatiques."""
    print("=" * 50)
    print(f"AUTO-UPDATE : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    news_updated = refresh_news()
    prices_updated = refresh_prices()
    history_updated = refresh_history()

    # Sentiment apres les news
    sentiment_updated = refresh_sentiment()

    print("=" * 50)
    return {
        "news": news_updated,
        "prices": prices_updated,
        "history": history_updated,
        "sentiment": sentiment_updated,
    }


if __name__ == "__main__":
    auto_update_all()