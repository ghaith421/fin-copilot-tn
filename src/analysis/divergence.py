import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DATA_RAW


def compute_divergence(window_days=7, min_news=3):
    """
    Detecte les divergences entre sentiment et cours.

    Divergence positive = sentiment > 0 mais cours < 0 (opportunite achat)
    Divergence negative = sentiment < 0 mais cours > 0 (opportunite vente)
    """
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
    except Exception as ex:
        print(f"[divergence] Erreur chargement : {ex}")
        return pd.DataFrame()

    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")
    sent_col = "sentiment" if "sentiment" in news.columns else "sentiment_kw"

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
    news_recent = news[news["date"] >= cutoff]
    prices_recent = prices[prices["date"] >= cutoff]

    # Sentiment moyen par banque sur la periode
    sent_by_bank = news_recent.groupby("tickers").agg(
        sentiment_moyen=(sent_col, "mean"),
        nb_news=(sent_col, "count"),
    ).round(3)

    # Variation du cours par banque sur la periode
    results = []
    for ticker in sent_by_bank.index:
        if ticker == "SECTEUR":
            continue

        p = prices_recent[prices_recent["ticker"] == ticker].sort_values("date")
        if len(p) < 2:
            continue

        first_price = p.iloc[0]["close"]
        last_price = p.iloc[-1]["close"]
        var_pct = (last_price - first_price) / first_price * 100

        sent = sent_by_bank.loc[ticker, "sentiment_moyen"]
        nb = int(sent_by_bank.loc[ticker, "nb_news"])

                # Si pas assez de news, afficher quand meme avec un statut
        if nb < min_news:
            results.append({
                "ticker": ticker,
                "sentiment": round(sent, 3) if nb > 0 else 0,
                "var_cours_pct": round(var_pct, 2),
                "nb_news": nb,
                "divergence": 0,
                "type": "Donnees insuffisantes",
                "emoji": "⚫",
                "description": f"Seulement {nb} news sur la periode",
            })
            continue

        # Divergence = sentiment - variation du cours (normalisees)
        sent_normalized = sent  # deja entre -1 et +1
        var_normalized = np.clip(var_pct / 10, -1, 1)  # 10% = 1.0

        divergence = sent_normalized - var_normalized

        # Classifier
        if divergence > 0.4 and sent > 0.1:
            kind = "Achat potentiel"
            emoji = "🟢"
            desc = "Sentiment positif mais cours pas encore monte"
        elif divergence < -0.4 and sent < -0.1:
            kind = "Vente potentielle"
            emoji = "🔴"
            desc = "Sentiment negatif mais cours pas encore baisse"
        else:
            kind = "Coherent"
            emoji = "⚪"
            desc = "Sentiment et cours alignes"

        results.append({
            "ticker": ticker,
            "sentiment": round(sent, 3),
            "var_cours_pct": round(var_pct, 2),
            "nb_news": nb,
            "divergence": round(divergence, 3),
            "type": kind,
            "emoji": emoji,
            "description": desc,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values("divergence", ascending=False).reset_index(drop=True)
    return df


def compute_divergence_history(ticker, window_days=7):
    """
    Calcule l'historique de la divergence pour une banque.
    Utile pour visualiser l'evolution.
    """
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
    except Exception:
        return pd.DataFrame()

    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")
    sent_col = "sentiment" if "sentiment" in news.columns else "sentiment_kw"

    news_t = news[news["tickers"] == ticker].copy()
    prices_t = prices[prices["ticker"] == ticker].sort_values("date").copy()

    if len(news_t) == 0 or len(prices_t) == 0:
        return pd.DataFrame()

    # Agregation par jour
    daily_sent = news_t.groupby("date")[sent_col].mean().reset_index()
    daily_sent.columns = ["date", "sentiment_jour"]

    # Fusionner avec les prix
    merged = prices_t.merge(daily_sent, on="date", how="left")
    merged["sentiment_jour"] = merged["sentiment_jour"].fillna(0)

    # Sentiment glissant
    merged["sentiment_rolling"] = merged["sentiment_jour"].rolling(window_days, min_periods=1).mean()

    # Variation du cours sur window_days
    merged["cours_var"] = merged["close"].pct_change(window_days) * 100

    # Divergence
    merged["cours_normalized"] = np.clip(merged["cours_var"] / 10, -1, 1)
    merged["divergence"] = merged["sentiment_rolling"] - merged["cours_normalized"]

    return merged[["date", "close", "sentiment_rolling", "cours_var", "divergence"]].dropna(subset=["divergence"])


if __name__ == "__main__":
    print("=== Signal Divergence — 7 jours ===\n")
    df = compute_divergence(window_days=7, min_news=2)
    if len(df) > 0:
        print(df[["ticker", "sentiment", "var_cours_pct", "nb_news", "divergence", "type"]].to_string(index=False))
    else:
        print("Aucune divergence detectee (pas assez de donnees)")

    print("\n=== Signal Divergence — 30 jours ===\n")
    df30 = compute_divergence(window_days=30, min_news=3)
    if len(df30) > 0:
        print(df30[["ticker", "sentiment", "var_cours_pct", "nb_news", "divergence", "type"]].to_string(index=False))