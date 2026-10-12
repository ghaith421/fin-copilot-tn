import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import json
import time
import re
import numpy as np
from groq import Groq
from config import DATA_RAW, GROQ_API_KEY

# =========================================================
# DICTIONNAIRES
# =========================================================
POSITIVE_WORDS = {
    "hausse": 1.0, "augmentation": 0.8, "croissance": 0.9, "progression": 0.9,
    "record": 1.0, "excellent": 1.0, "performant": 0.9, "amélioration": 0.8,
    "amelioration": 0.8, "accélération": 0.8, "acceleration": 0.8,
    "bénéfice": 1.0, "benefice": 1.0, "profit": 0.9, "gains": 0.8,
    "dividende": 0.8, "rendement": 0.6, "rentabilité": 0.9, "rentabilite": 0.9,
    "expansion": 0.8, "investissement": 0.5, "partenariat": 0.6,
    "acquisition": 0.7, "nouveau": 0.4, "lancement": 0.5,
    "succès": 0.9, "succes": 0.9, "réussi": 0.8, "reussi": 0.8,
    "positif": 0.7, "favorable": 0.7, "optimiste": 0.7,
    "surperforme": 0.8, "solide": 0.7, "robuste": 0.6,
    "rebond": 0.8, "appréciation": 0.7, "appreciation": 0.7,
}

NEGATIVE_WORDS = {
    "baisse": -1.0, "diminution": -0.8, "chute": -1.0, "effondrement": -1.0,
    "perte": -1.0, "pertes": -1.0, "déficit": -0.9, "deficit": -0.9,
    "recul": -0.8, "dégradation": -0.9, "degradation": -0.9,
    "amende": -0.9, "sanction": -0.9, "pénalité": -0.9, "penalite": -0.9,
    "condamnation": -1.0, "enquête": -0.7, "enquete": -0.7, "fraude": -1.0,
    "scandale": -1.0, "litige": -0.7, "contentieux": -0.7,
    "difficulté": -0.8, "difficulte": -0.8, "crise": -0.9,
    "problème": -0.7, "probleme": -0.7, "risque": -0.6,
    "avertissement": -0.7, "alerte": -0.6,
    "négatif": -0.7, "negatif": -0.7, "défavorable": -0.7, "defavorable": -0.7,
    "pessimiste": -0.7, "inquiétude": -0.7, "inquietude": -0.7,
    "fermeture": -0.7, "licenciement": -0.8, "restructuration": -0.5,
    "sous-performe": -0.7, "faiblesse": -0.7, "fragile": -0.6,
}

NEGATIONS = ["pas", "non", "aucun", "aucune", "sans", "jamais", "ni",
             "ne", "n'", "point", "guère", "guere"]

EVENT_PATTERNS = {
    "resultats": r"(résultat|resultat|résultats|resultats|bénéfice|benefice|chiffre d'affaires|pnb|bilan)",
    "dividende": r"(dividende|distribution|assemblee generale|assemblée générale|AGO|AG)",
    "regulation": r"(BCT|banque centrale|régulation|regulation|loi|décret|decret|amende|sanction)",
    "macro": r"(inflation|taux|BCE|FED|économie|economie|croissance du PIB|chômage|chomage)",
    "fusion": r"(fusion|acquisition|OPA|partenariat|rachat)",
    "direction": r"(directeur|PDG|DG|président|president|nomination|démission|demission)",
}

SOURCE_WEIGHTS = {
    "reuters": 1.5, "bloomberg": 1.5, "afp": 1.5,
    "ilboursa": 1.2, "tustex": 1.2, "lapresse": 1.2, "businessnews": 1.2,
    "yahoo": 1.0, "cnbc": 1.0, "marketwatch": 1.0,
    "google_news": 1.0, "investing": 1.0,
    "seekingalpha": 0.9, "reddit": 0.5,
}


# =========================================================
# FONCTIONS
# =========================================================
def keyword_sentiment(text):
    """Sentiment par mots-cles avec detection de negation."""
    text_lower = (text or "").lower()
    words = re.findall(r"\b\w+\b", text_lower)

    if not words:
        return 0.0

    score = 0.0
    matches = 0

    for i, word in enumerate(words):
        weight = 0.0
        if word in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[word]
        elif word in NEGATIVE_WORDS:
            weight = NEGATIVE_WORDS[word]
        else:
            continue

        context = words[max(0, i - 4):i]
        has_negation = any(neg in " ".join(context) for neg in NEGATIONS)

        if has_negation:
            weight = -weight

        score += weight
        matches += 1

    if matches == 0:
        return 0.0

    normalized = score / (matches ** 0.5 + 1)
    return max(-1.0, min(1.0, normalized))


def detect_event(text):
    for event, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, text or "", re.IGNORECASE):
            return event
    return "autre"


def impact_from_score(score):
    if score > 0.2:
        return "haussier"
    elif score < -0.2:
        return "baissier"
    return "neutre"


def source_weight(source):
    if not source:
        return 1.0
    s = str(source).lower()
    for key, w in SOURCE_WEIGHTS.items():
        if key in s:
            return w
    return 0.8


def recency_weight(date_val):
    """Poids de recence (1.0 aujourd'hui, decroit sur 60j)."""
    try:
        d = pd.to_datetime(date_val).tz_localize(None)
        age_days = (pd.Timestamp.now() - d).days
        if age_days < 0:
            return 1.0
        # Decroissance plus douce sur 60 jours (au lieu de 30)
        return float(max(0.3, np.exp(-age_days / 60.0)))
    except Exception:
        return 1.0


MODEL_LLAMA = "openai/gpt-oss-20b"
SLEEP_LLAMA = 4.5


def llama_sentiment(client, title, summary=""):
    text = f"Titre : {title}\nResume : {summary[:200]}"
    prompt = """Analyse cette news bancaire tunisienne. Reponds UNIQUEMENT en JSON :
{"sentiment": -1.0, "raison": "explication courte"}"""
    try:
        r = client.chat.completions.create(
            model=MODEL_LLAMA,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        data = json.loads(r.choices[0].message.content)
        return float(data.get("sentiment", 0)), data.get("raison", "")[:150]
    except Exception as ex:
        return None, str(ex)[:80]


# =========================================================
# PIPELINE
# =========================================================
def score_news_keywords_only():
    """Version rapide : mots-cles + poids stockes separement."""
    path = f"{DATA_RAW}/banks_news.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[sentiment-kw] {len(df)} news a analyser")

    kw_sentiments = []
    events = []
    for _, row in df.iterrows():
        text = (row["title"] or "") + " " + (row.get("summary", "") or "")
        kw_sentiments.append(keyword_sentiment(text))
        events.append(detect_event(text))

    df["sentiment_kw"] = kw_sentiments
    df["event_type"] = events

    # Les poids sont STOCKES mais PAS appliques au sentiment individuel
    df["source_weight"] = df["source"].apply(source_weight) if "source" in df.columns else 1.0
    df["recency_weight"] = df["date"].apply(recency_weight) if "date" in df.columns else 1.0

    # Le sentiment individuel reste BRUT (avec negation)
    df["sentiment"] = df["sentiment_kw"]
    df["impact"] = df["sentiment"].apply(impact_from_score)

    out = f"{DATA_RAW}/banks_news_scored.parquet"
    df.to_parquet(out)
    print(f"[sentiment-kw] Sauvegarde -> {out}")
    print(f"[sentiment-kw] Sentiment moyen : {df['sentiment'].mean():+.3f}")

    print("\nSentiment moyen pondere par banque (avec source + recence) :")
    for bank in df["tickers"].unique():
        sub = df[df["tickers"] == bank]
        if len(sub) == 0:
            continue
        weights = sub["source_weight"] * sub["recency_weight"]
        if weights.sum() > 0:
            weighted_avg = (sub["sentiment"] * weights).sum() / weights.sum()
        else:
            weighted_avg = sub["sentiment"].mean()
        print(f"  {bank} : {weighted_avg:+.3f}  (simple: {sub['sentiment'].mean():+.3f}, n={len(sub)})")

    return df


def score_news():
    path = f"{DATA_RAW}/banks_news.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[sentiment] {len(df)} news a analyser")

    print("\n[sentiment] === ETAPE 1 : Mots-cles + negation ===")
    kw_sentiments = []
    events = []
    for _, row in df.iterrows():
        text = (row["title"] or "") + " " + (row.get("summary", "") or "")
        kw_sentiments.append(keyword_sentiment(text))
        events.append(detect_event(text))

    df["sentiment_kw"] = kw_sentiments
    df["event_type"] = events
    df["source_weight"] = df["source"].apply(source_weight) if "source" in df.columns else 1.0
    df["recency_weight"] = df["date"].apply(recency_weight) if "date" in df.columns else 1.0

    print(f"  Positifs (kw > +0.2) : {(df['sentiment_kw'] > 0.2).sum()}")
    print(f"  Negatifs (kw < -0.2) : {(df['sentiment_kw'] < -0.2).sum()}")

    print("\n[sentiment] === ETAPE 2 : Llama sur les news fortes ===")
    strong_mask = df["sentiment_kw"].abs() > 0.3
    strong_df = df[strong_mask].copy()
    print(f"  {len(strong_df)} news fortes a verifier par Llama")

    if len(strong_df) > 0 and GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        llama_scores = []
        llama_raisons = []
        for i, (_, row) in enumerate(strong_df.iterrows()):
            score, raison = llama_sentiment(client, row["title"], row.get("summary", ""))
            llama_scores.append(score)
            llama_raisons.append(raison)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(strong_df)} validees")
            time.sleep(SLEEP_LLAMA)

        strong_df["sentiment_llama"] = llama_scores
        strong_df["raison"] = llama_raisons
        df = df.merge(strong_df[["title", "sentiment_llama", "raison"]], on="title", how="left")
    else:
        df["sentiment_llama"] = None
        df["raison"] = ""

    # Sentiment individuel : kw + llama, SANS poids
    def final_score(row):
        kw = row["sentiment_kw"]
        ll = row.get("sentiment_llama")
        if pd.notna(ll):
            return float(np.clip(0.4 * kw + 0.6 * ll, -1, 1))
        return float(np.clip(kw, -1, 1))

    df["sentiment"] = df.apply(final_score, axis=1)
    df["impact"] = df["sentiment"].apply(impact_from_score)

    out = f"{DATA_RAW}/banks_news_scored.parquet"
    df.to_parquet(out)
    print(f"\n[sentiment] Sauvegarde -> {out}")

    print("\n" + "=" * 50)
    print("STATISTIQUES")
    print("=" * 50)
    print(f"Sentiment moyen : {df['sentiment'].mean():+.3f}")
    print("\nPar banque :")
    print(df.groupby("tickers")["sentiment"].agg(["mean", "count"]).round(3))
    return df


if __name__ == "__main__":
    if "--keywords-only" in sys.argv:
        score_news_keywords_only()
    else:
        score_news()