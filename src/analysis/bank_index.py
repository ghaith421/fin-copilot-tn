import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_RAW


def build_bank_index():
    """
    Construit un indice bancaire synthetique a partir des 12 banques.
    Methode : moyenne des rendements journaliers (equiponderee).
    """
    try:
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
    except Exception:
        print("[index] Aucun historique de prix")
        return pd.DataFrame()

    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Rendement journalier par banque
    prices["daily_ret"] = prices.groupby("ticker")["close"].pct_change()

    # Moyenne equiponderee des rendements (par date)
    index_df = prices.groupby("date")["daily_ret"].mean().reset_index()
    index_df = index_df.dropna()
    index_df = index_df.rename(columns={"daily_ret": "index_ret"})

    # Cumul de l'indice (base 1000 au debut)
    index_df = index_df.sort_values("date").reset_index(drop=True)
    index_df["index_level"] = 1000 * (1 + index_df["index_ret"]).cumprod()

    out = f"{DATA_RAW}/bank_index.parquet"
    index_df.to_parquet(out)

    print(f"[index] {len(index_df)} jours d'indice bancaire -> {out}")
    print(f"[index] Periode : {index_df['date'].min().date()} -> {index_df['date'].max().date()}")
    print(f"[index] Niveau initial : 1000")
    print(f"[index] Niveau final   : {index_df['index_level'].iloc[-1]:.0f}")
    print(f"[index] Rendement total : {(index_df['index_level'].iloc[-1]/1000 - 1)*100:+.2f}%")

    return index_df


def compute_benchmark_returns(index_df, start_date, horizons=(1, 5, 20)):
    """
    Calcule le rendement de l'indice sur J+1, J+5, J+20 a partir d'une date.
    """
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


if __name__ == "__main__":
    index_df = build_bank_index()