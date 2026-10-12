import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from scipy import stats
from config import DATA_RAW

# =========================================================
# REGRESSION : SENTIMENT -> RENDEMENT DU COURS
# =========================================================


def prepare_dataset():
    """
    Pour chaque news, calcule le rendement du cours a J+1, J+5, J+20
    apres la publication.
    """
    # Charger les news
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news.parquet")

    # Charger les prix historiques
    try:
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
    except Exception:
        print("[reg] Aucun historique de prix disponible")
        return pd.DataFrame()

    # Normaliser les dates
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None).dt.floor("D")
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).dt.floor("D")

    sent_col = "sentiment" if "sentiment" in news.columns else "sentiment_kw"

    # Calcul des rendements futurs par (ticker, date)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    for h in [1, 5, 20]:
        prices[f"ret_{h}"] = prices.groupby("ticker")["close"].transform(
            lambda s: s.shift(-h) / s - 1
        )

    # Pour chaque news, trouver le prochain jour de trading
    rows = []
    for _, n in news.iterrows():
        ticker = n["tickers"]
        ndate = n["date"]

        sub = prices[prices["ticker"] == ticker]
        if len(sub) == 0:
            continue

        future = sub[sub["date"] >= ndate]
        if len(future) == 0:
            continue

        entry = future.iloc[0]
        rows.append({
            "date": entry["date"],
            "ticker": ticker,
            "sentiment": n[sent_col],
            "event_type": n.get("event_type", "autre"),
            "ret_1": entry.get("ret_1", np.nan),
            "ret_5": entry.get("ret_5", np.nan),
            "ret_20": entry.get("ret_20", np.nan),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["sentiment"])
    return df


def run_regression(df, horizon=5):
    """Regression lineaire : ret_J+h ~ sentiment."""
    target = f"ret_{horizon}"
    sub = df.dropna(subset=[target, "sentiment"])

    if len(sub) < 20:
        return {
            "error": f"Pas assez de donnees (n={len(sub)}). Il faut au moins 20 observations.",
            "n": len(sub),
        }

    x = sub["sentiment"].values
    y = sub[target].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "n": len(sub),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "r": r_value,
        "p_value": p_value,
        "std_err": std_err,
        "significant": p_value < 0.05,
        "horizon": horizon,
    }


def run_all_regressions(df):
    """Lance les regressions sur J+1, J+5, J+20."""
    results = {}
    for h in [1, 5, 20]:
        results[f"J+{h}"] = run_regression(df, horizon=h)
    return results


def analyze_by_event(df):
    """Analyse l'impact par type d'evenement."""
    rows = []
    for evt in df["event_type"].dropna().unique():
        sub = df[df["event_type"] == evt]
        if len(sub) < 5:
            continue
        rows.append({
            "event": evt,
            "n": len(sub),
            "sentiment_moyen": sub["sentiment"].mean(),
            "ret_1_moyen": sub["ret_1"].mean() if "ret_1" in sub.columns else None,
            "ret_5_moyen": sub["ret_5"].mean() if "ret_5" in sub.columns else None,
            "ret_20_moyen": sub["ret_20"].mean() if "ret_20" in sub.columns else None,
        })
    return pd.DataFrame(rows).sort_values("ret_5_moyen", ascending=False)


def format_results(results):
    """Formate les resultats en texte lisible."""
    lines = ["=" * 60]
    lines.append("REGRESSION : SENTIMENT -> RENDEMENT DU COURS")
    lines.append("=" * 60)

    for horizon, r in results.items():
        lines.append(f"\n--- {horizon} ---")
        if "error" in r:
            lines.append(f"  {r['error']}")
            continue

        lines.append(f"  Observations : {r['n']}")
        lines.append(f"  Coefficient (beta) : {r['slope']:.6f}")
        lines.append(f"  R carre : {r['r_squared']:.4f}")
        lines.append(f"  Correlation (r) : {r['r']:.4f}")
        lines.append(f"  P-value : {r['p_value']:.4f}")
        lines.append(f"  Significatif (p<0.05) : {'OUI' if r['significant'] else 'NON'}")

        if r["significant"]:
            direction = "positif" if r["slope"] > 0 else "negatif"
            lines.append(f"\n  Interpretation :")
            lines.append(f"  Une hausse de +1.0 de sentiment est associee a un rendement")
            lines.append(f"  {direction} de {r['slope']*100:.2f}% sur {horizon}.")
        else:
            lines.append(f"\n  Pas de relation statistiquement significative.")

    return "\n".join(lines)


if __name__ == "__main__":
    print("[reg] Preparation des donnees...")
    df = prepare_dataset()

    if len(df) == 0:
        print("[reg] Aucune donnee. Il faut des prix historiques.")
    else:
        print(f"[reg] {len(df)} observations pretes")
        results = run_all_regressions(df)
        print(format_results(results))

        print("\n" + "=" * 60)
        print("ANALYSE PAR TYPE D'EVENEMENT")
        print("=" * 60)
        evt_df = analyze_by_event(df)
        print(evt_df.to_string(index=False))