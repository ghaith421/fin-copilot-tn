import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_RAW


def build_news_impact():
    """
    Pour chaque news, associe :
    - Le prix au moment de la news
    - Le prix a J+1, J+5, J+20
    - Le % de variation
    - Un verdict : le sentiment a-t-il ete confirme par le cours ?
    """
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news.parquet")

    try:
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
    except Exception:
        print("[impact] Aucun historique de prix")
        return pd.DataFrame()

    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")

    sent_col = "sentiment" if "sentiment" in news.columns else "sentiment_kw"

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    rows = []
    for _, n in news.iterrows():
        ticker = n["tickers"]
        ndate = n["date"]

        sub = prices[prices["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if len(sub) == 0:
            continue

        idx_candidates = sub[sub["date"] >= ndate].index
        if len(idx_candidates) == 0:
            continue
        i = idx_candidates[0]

        entry = sub.iloc[i]
        price_at_news = entry["close"]

        def get_price(offset):
            j = i + offset
            if j < len(sub):
                return sub.iloc[j]["close"]
            return np.nan

        p1 = get_price(1)
        p5 = get_price(5)
        p20 = get_price(20)

        def pct(p):
            if pd.notna(p) and price_at_news > 0:
                return (p - price_at_news) / price_at_news * 100
            return np.nan

        var1 = pct(p1)
        var5 = pct(p5)
        var20 = pct(p20)

        sentiment = n[sent_col]

        verdict = "En attente"
        if pd.notna(var20):
            if sentiment > 0.2 and var20 > 0:
                verdict = "Confirme (positif -> hausse)"
            elif sentiment > 0.2 and var20 < 0:
                verdict = "Non confirme (positif -> baisse)"
            elif sentiment < -0.2 and var20 < 0:
                verdict = "Confirme (negatif -> baisse)"
            elif sentiment < -0.2 and var20 > 0:
                verdict = "Non confirme (negatif -> hausse)"
            else:
                verdict = "Neutre"

        rows.append({
            "date": ndate,
            "ticker": ticker,
            "title": n["title"],
            "source": n.get("source", ""),
            "event_type": n.get("event_type", ""),
            "sentiment": sentiment,
            "price_at_news": price_at_news,
            "price_j1": p1,
            "price_j5": p5,
            "price_j20": p20,
            "var_j1_pct": var1,
            "var_j5_pct": var5,
            "var_j20_pct": var20,
            "verdict": verdict,
            "url": n.get("url", ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    out = f"{DATA_RAW}/news_impact.parquet"
    df.to_parquet(out)

    n_confirmed = df["verdict"].str.contains("Confirme", na=False).sum()
    n_with_j20 = df["var_j20_pct"].notna().sum()
    print(f"[impact] {len(df)} news avec impact calcule -> {out}")
    print(f"[impact] {n_with_j20} news ont un J+20 mesurable")
    print(f"[impact] {n_confirmed} news ou le sentiment a ete confirme par le cours")

    return df


def get_news_for_bank(df, ticker, only_with_impact=False):
    sub = df[df["ticker"] == ticker].copy()
    if only_with_impact:
        sub = sub.dropna(subset=["var_j20_pct"])
    return sub.sort_values("date", ascending=False)


def get_impact_stats(df, ticker=None):
    if ticker:
        df = df[df["ticker"] == ticker]

    df_valid = df.dropna(subset=["var_j20_pct"])
    if len(df_valid) == 0:
        return None

    pos = df_valid[df_valid["sentiment"] > 0.2]
    neg = df_valid[df_valid["sentiment"] < -0.2]

    return {
        "n_total": len(df_valid),
        "positif": {
            "n": len(pos),
            "var_j1": pos["var_j1_pct"].mean() if len(pos) > 0 else None,
            "var_j5": pos["var_j5_pct"].mean() if len(pos) > 0 else None,
            "var_j20": pos["var_j20_pct"].mean() if len(pos) > 0 else None,
            "hit_rate": (pos["var_j20_pct"] > 0).mean() * 100 if len(pos) > 0 else None,
        },
        "negatif": {
            "n": len(neg),
            "var_j1": neg["var_j1_pct"].mean() if len(neg) > 0 else None,
            "var_j5": neg["var_j5_pct"].mean() if len(neg) > 0 else None,
            "var_j20": neg["var_j20_pct"].mean() if len(neg) > 0 else None,
            "hit_rate": (neg["var_j20_pct"] < 0).mean() * 100 if len(neg) > 0 else None,
        },
    }


if __name__ == "__main__":
    df = build_news_impact()

    if len(df) > 0:
        print("\n=== STATS GLOBALES ===")
        stats = get_impact_stats(df)
        if stats:
            print(f"Total : {stats['n_total']} news avec J+20 mesurable")
            print(f"\nSentiment POSITIF :")
            p = stats['positif']
            print(f"  n = {p['n']}")
            if p['var_j20'] is not None:
                print(f"  Rendement moyen J+1  : {p['var_j1']:+.2f}%")
                print(f"  Rendement moyen J+5  : {p['var_j5']:+.2f}%")
                print(f"  Rendement moyen J+20 : {p['var_j20']:+.2f}%")
                print(f"  Taux de hausse J+20  : {p['hit_rate']:.1f}%")
            print(f"\nSentiment NEGATIF :")
            ng = stats['negatif']
            print(f"  n = {ng['n']}")
            if ng['var_j20'] is not None:
                print(f"  Rendement moyen J+1  : {ng['var_j1']:+.2f}%")
                print(f"  Rendement moyen J+5  : {ng['var_j5']:+.2f}%")
                print(f"  Rendement moyen J+20 : {ng['var_j20']:+.2f}%")
                print(f"  Taux de baisse J+20  : {ng['hit_rate']:.1f}%")

        # Exemples de news anciennes avec impact
        print("\n=== EXEMPLES DE NEWS AVEC IMPACT MESURABLE ===")
        df_with = df.dropna(subset=["var_j20_pct"]).sort_values("date")
        for ticker in ["BIAT", "BH", "ATB", "UBCI"]:
            sub = df_with[df_with["ticker"] == ticker].tail(2)
            if len(sub) == 0:
                continue
            print(f"\n--- {ticker} ---")
            for _, r in sub.iterrows():
                print(f"  [{r['date'].date()}] {r['title'][:70]}")
                print(f"     Sentiment {r['sentiment']:+.2f} | Prix {r['price_at_news']:.2f} -> {r['price_j20']:.2f} ({r['var_j20_pct']:+.2f}%) | {r['verdict']}")