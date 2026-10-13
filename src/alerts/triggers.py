import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from datetime import datetime, timedelta
from config import DATA_RAW


def load_news():
    try:
        return pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        return pd.DataFrame()


def get_last_sent_ids():
    """Recupere les IDs (titles) des news deja notifiees."""
    path = f"{DATA_RAW}/notified_news.txt"
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception:
        return set()


def save_notified_ids(ids):
    """Sauvegarde les IDs notifies."""
    path = f"{DATA_RAW}/notified_news.txt"
    with open(path, "w", encoding="utf-8") as f:
        for i in ids:
            f.write(f"{i}\n")


def send_email_for_news(news_item):
    """Envoie un email pour une news."""
    try:
        from src.alerts.email_sender import send_news_alert
        return send_news_alert(news_item)
    except Exception as ex:
        print(f"[trigger] Erreur envoi email : {ex}")
        return False


def notify_new_news(max_emails=20, only_strong=False):
    """
    Envoie un email pour chaque NOUVELLE news.

    Args:
        max_emails : nombre max d'emails par run (pour eviter le spam)
        only_strong : si True, envoie seulement les news |sentiment| > 0.3
    """
    df = load_news()
    if len(df) == 0:
        print("[trigger] Aucune news disponible")
        return

    # Charger l'historique des news deja notifiees
    notified = get_last_sent_ids()
    print(f"[trigger] {len(notified)} news deja notifiees")

    # Trouver les nouvelles news
    df["_id"] = df["title"].fillna("").str.strip()
    new_news = df[~df["_id"].isin(notified)].copy()

    # Trier par date desc (les plus recentes d'abord)
    new_news = new_news.sort_values("date", ascending=False)

    # Optionnel : filtrer les news fortes
    if only_strong:
        new_news = new_news[new_news["sentiment"].abs() > 0.3]

    n_total = len(new_news)
    print(f"[trigger] {n_total} nouvelles news a notifier")

    if n_total == 0:
        return

    # Limiter le nombre d'emails par run
    to_send = new_news.head(max_emails)
    print(f"[trigger] On envoie {len(to_send)} emails (max {max_emails})")

    sent_count = 0
    new_notified = set(notified)

    for _, row in to_send.iterrows():
        news_item = {
            "tickers": row.get("tickers", "?"),
            "title": row.get("title", ""),
            "sentiment": row.get("sentiment", 0),
            "event_type": row.get("event_type", "autre"),
            "impact": row.get("impact", "neutre"),
            "raison": row.get("raison", ""),
            "source": row.get("source", ""),
            "date": str(row.get("date", "")),
            "url": row.get("url", ""),
        }

        ok = send_email_for_news(news_item)
        if ok:
            new_notified.add(row["_id"])
            sent_count += 1
        else:
            print(f"[trigger] Echec envoi : {row['title'][:60]}")

    # Sauvegarder l'historique
    save_notified_ids(new_notified)
    print(f"[trigger] {sent_count} emails envoyes")


if __name__ == "__main__":
    # Mode test : n'envoie que les 3 premieres
    import sys
    if "--test" in sys.argv:
        notify_new_news(max_emails=3)
    else:
        notify_new_news(max_emails=20)