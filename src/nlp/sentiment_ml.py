import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_RAW

# =========================================================
# MODELE ML SPECIALISE FINANCE (francais)
# =========================================================
MODEL_NAME = "bardsai/finance-sentiment-fr-base"

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print(f"[ml] Chargement du modele {MODEL_NAME}...")
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            truncation=True,
            max_length=512,
            top_k=None, # Retourne toutes les classes avec leurs probabilites
        )
        print("[ml] Modele charge.")
    return _pipeline


def score_text_ml(text):
    """
    Analyse un texte complet et retourne un score -1 a +1.
    Le modele bardsai retourne : positive / negative / neutral
    """
    if not text or len(text.strip()) < 10:
        return {"score": 0.0, "confidence": 0.0, "label": "neutral"}

    pipe = get_pipeline()
    text = text[:1000] # Tronquer pour eviter les erreurs

    try:
        results = pipe(text)[0]
        best = max(results, key=lambda x: x["score"])
        label = best["label"].lower()
        confidence = best["score"]

        # Mapping des labels vers score
        if "positive" in label:
            score = 1.0
        elif "negative" in label:
            score = -1.0
        else:
            score = 0.0

        # Ajuster avec la confiance (ex: si confiance 0.6 -> score 0.6, pas 1.0)
        final_score = score * confidence

        return {
            "score": round(final_score, 3),
            "confidence": round(confidence, 3),
            "label": label,
        }
    except Exception as ex:
        print(f"[ml] Erreur : {str(ex)[:100]}")
        return {"score": 0.0, "confidence": 0.0, "label": "error"}


def score_news_ml():
    """Analyse toutes les news avec le modele finance."""
    path = f"{DATA_RAW}/banks_news.parquet"
    if not os.path.exists(path):
        print(f"[ml] Fichier manquant : {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[ml] {len(df)} news a analyser")

    scores = []
    confidences = []
    labels = []

    for i, row in df.iterrows():
        title = row.get("title", "") or ""
        summary = row.get("summary", "") or ""
        text = f"{title}. {summary}"

        result = score_text_ml(text)
        scores.append(result["score"])
        confidences.append(result["confidence"])
        labels.append(result["label"])

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} analysees")

    df["sentiment_ml"] = scores
    df["sentiment_confidence"] = confidences
    df["sentiment_label"] = labels
    df["sentiment"] = df["sentiment_ml"]
    df["impact"] = df["sentiment"].apply(
        lambda s: "haussier" if s > 0.2 else ("baissier" if s < -0.2 else "neutre")
    )

    out = f"{DATA_RAW}/banks_news_scored.parquet"
    df.to_parquet(out)
    print(f"\n[ml] Sauvegarde -> {out}")
    print(f"[ml] Sentiment moyen : {df['sentiment'].mean():+.3f}")
    print(f"[ml] News avec sentiment != 0 : {(df['sentiment'] != 0).sum()} / {len(df)}")

    print("\nPar banque :")
    print(df.groupby("tickers")["sentiment"].agg(["mean", "count"]).round(3))

    return df


if __name__ == "__main__":
    tests = [
        "BIAT annonce un PNB en hausse de 15% au premier semestre",
        "STB : amende de 5 millions de dinars suite a une fraude",
        "BH Bank : pas de benefice cette annee, deficit de 20 MTND",
        "UBCI : solidite financiere confirmee, croissance soutenue",
        "ATB : resultats en baisse, mais la direction reste optimiste pour 2025",
    ]

    print("=== TEST DU MODELE FINANCE ===\n")
    for t in tests:
        result = score_text_ml(t)
        print(f"News : {t[:70]}...")
        print(f"  Score      : {result['score']:+.2f}")
        print(f"  Confiance  : {result['confidence']:.2%}")
        print(f"  Label      : {result['label']}")
        print()