import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_RAW


def _compute_bench(index_df, start_date, horizons=(1, 5, 20)):
    """Calcule le rendement de l'indice bancaire a J+h."""
    index_df = index_df.sort_values("date").reset_index(drop=True)
    idx = index_df[index_df["date"] >= start_date].index
    if len(idx) == 0:
        return {f"bench_j{h}": np.nan for h in horizons}
    i = idx[0]
    base = index_df.iloc[i]["index_level"]
    result = {}
    for h in horizons:
        j = i + h
        if j < len(index_df):
            result[f"bench_j{h}"] = (index_df.iloc[j]["index_level"] / base - 1) * 100
        else:
            result[f"bench_j{h}"] = np.nan
    return result


def build_clean_impact():
    """
    Version nettoyee avec benchmark :
    - Dedup par (ticker, date)
    - Detection des dividendes
    - Comparaison vs indice bancaire
    """
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        return pd.DataFrame()

    try:
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
    except Exception:
        return pd.DataFrame()

    try:
        index_df = pd.read_parquet(f"{DATA_RAW}/bank_index.parquet")
        index_df["date"] = pd.to_datetime(index_df["date"])
    except Exception:
        index_df = None
        print("[impact] Pas d'indice bancaire (lance: python -m src.analysis.bank_index)")

    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")
    sent_col = "sentiment" if "sentiment" in news.columns else "sentiment_kw"

    news["abs_sent"] = news[sent_col].abs()
    news = news.sort_values("abs_sent", ascending=False)
    news = news.drop_duplicates(subset=["tickers", "date"], keep="first")
    news = news.drop(columns=["abs_sent"])

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["daily_ret"] = prices.groupby("ticker")["close"].pct_change()
    div_dates = prices[prices["daily_ret"] < -0.15][["ticker", "date"]].copy()
    div_set = set(zip(div_dates["ticker"], div_dates["date"]))

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

        window = sub.iloc[i:i+21]
        has_dividend = any((t, d) in div_set for t, d in zip(window["ticker"], window["date"]))

        sentiment = n[sent_col]

        # Benchmark
        bench = {}
        if index_df is not None:
            bench = _compute_bench(index_df, entry["date"])

        bench_j1 = bench.get("bench_j1", np.nan)
        bench_j5 = bench.get("bench_j5", np.nan)
        bench_j20 = bench.get("bench_j20", np.nan)

        alpha_j1 = var1 - bench_j1 if pd.notna(var1) and pd.notna(bench_j1) else np.nan
        alpha_j5 = var5 - bench_j5 if pd.notna(var5) and pd.notna(bench_j5) else np.nan
        alpha_j20 = var20 - bench_j20 if pd.notna(var20) and pd.notna(bench_j20) else np.nan

        verdict = "En attente"
        if pd.notna(var20):
            if has_dividend:
                verdict = "Dividende detecte"
            elif sentiment > 0.1 and var20 > 0:
                verdict = "Confirme (positif -> hausse)"
            elif sentiment > 0.1 and var20 < 0:
                verdict = "Non confirme (positif -> baisse)"
            elif sentiment < -0.1 and var20 < 0:
                verdict = "Confirme (negatif -> baisse)"
            elif sentiment < -0.1 and var20 > 0:
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
            "bench_j1_pct": bench_j1,
            "bench_j5_pct": bench_j5,
            "bench_j20_pct": bench_j20,
            "alpha_j1_pct": alpha_j1,
            "alpha_j5_pct": alpha_j5,
            "alpha_j20_pct": alpha_j20,
            "verdict": verdict,
            "has_dividend": has_dividend,
            "url": n.get("url", ""),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    out = f"{DATA_RAW}/news_impact_clean.parquet"
    df.to_parquet(out)

    # Stats
    df_clean = df[(df["has_dividend"] == False) & (df["var_j20_pct"].notna())]
    df_pos = df_clean[df_clean["sentiment"] > 0.1]
    df_neg = df_clean[df_clean["sentiment"] < -0.1]

    print(f"[impact] {len(df)} news traitees")
    print(f"[impact] {len(df_clean)} news avec impact mesurable (hors dividendes)")

    if len(df_pos) > 0:
        print(f"\n>>> SENTIMENT POSITIF (n={len(df_pos)}) :")
        print(f"    Rendement J+20 : {df_pos['var_j20_pct'].mean():+.2f}%")
        print(f"    Indice J+20    : {df_pos['bench_j20_pct'].mean():+.2f}%")
        print(f"    Alpha J+20     : {df_pos['alpha_j20_pct'].mean():+.2f} pp")

    if len(df_neg) > 0:
        print(f"\n>>> SENTIMENT NEGATIF (n={len(df_neg)}) :")
        print(f"    Rendement J+20 : {df_neg['var_j20_pct'].mean():+.2f}%")
        print(f"    Indice J+20    : {df_neg['bench_j20_pct'].mean():+.2f}%")
        print(f"    Alpha J+20     : {df_neg['alpha_j20_pct'].mean():+.2f} pp")

    return df


if __name__ == "__main__":
    build_clean_impact()