import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import json
import time
import re
from groq import Groq
from config import DATA_RAW, GROQ_API_KEY

# =========================================================
# 1) DICTIONNAIRES DE MOTS-CLES FINANCIERS
# =========================================================
POSITIVE_WORDS = {
    # Croissance / performance
    "hausse": 1.0, "augmentation": 0.8, "croissance": 0.9, "progression": 0.9,
    "record": 1.0, "excellent": 1.0, "performant": 0.9, "amélioration": 0.8,
    "amelioration": 0.8, "accélération": 0.8, "acceleration": 0.8,
    # Résultats
    "bénéfice": 1.0, "benefice": 1.0, "profit": 0.9, "gains": 0.8,
    "dividende": 0.8, "rendement": 0.6, "rentabilité": 0.9, "rentabilite": 0.9,
    # Expansion
    "expansion": 0.8, "investissement": 0.5, "partenariat": 0.6,
    "acquisition": 0.7, "nouveau": 0.4, "lancement": 0.5,
    # Succès
    "succès": 0.9, "succes": 0.9, "réussi": 0.8, "reussi": 0.8,
    "positif": 0.7, "favorable": 0.7, "optimiste": 0.7,
    "surperforme": 0.8, "solide": 0.7, "robuste": 0.6,
    # Prix
    "rebond": 0.8, "appréciation": 0.7, "appreciation": 0.7,
}

NEGATIVE_WORDS = {
    # Baisse / pertes
    "baisse": -1.0, "diminution": -0.8, "chute": -1.0, "effondrement": -1.0,
    "perte": -1.0, "pertes": -1.0, "déficit": -0.9, "deficit": -0.9,
    "recul": -0.8, "dégradation": -0.9, "degradation": -0.9,
    # Sanctions
    "amende": -0.9, "sanction": -0.9, "pénalité": -0.9, "penalite": -0.9,
    "condamnation": -1.0, "enquête": -0.7, "enquete": -0.7, "fraude": -1.0,
    "scandale": -1.0, "litige": -0.7, "contentieux": -0.7,
    # Difficultés
    "difficulté": -0.8, "difficulte": -0.8, "crise": -0.9,
    "problème": -0.7, "probleme": -0.7, "risque": -0.6,
    "avertissement": -0.7, "alerte": -0.6,
    # Négatif
    "négatif": -0.7, "negatif": -0.7, "défavorable": -0.7, "defavorable": -0.7,
    "pessimiste": -0.7, "inquiétude": -0.7, "inquietude": -0.7,
    # Autres
    "fermeture": -0.7, "licenciement": -0.8, "restructuration": -0.5,
    "sous-performe": -0.7, "faiblesse": -0.7, "fragile": -0.6,
}

# Types d'evenements (regex)
EVENT_PATTERNS = {
    "resultats": r"(résultat|resultat|résultats|resultats|bénéfice|benefice|chiffre d'affaires|pnb|bilan)",
    "dividende": r"(dividende|distribution|assemblee generale|assemblée générale|AGO|AG)",
    "regulation": r"(BCT|banque centrale|régulation|regulation|loi|décret|decret|amende|sanction)",
    "macro": r"(inflation|taux|BCE|FED|économie|economie|croissance du PIB|chômage|chomage)",
    "fusion": r"(fusion|acquisition|OPA|partenariat|rachat)",
    "direction": r"(directeur|PDG|DG|président|president|nomination|démission|demission)",
}


def keyword_sentiment(text):
    """Calcule un sentiment par mots-cles."""
    text_lower = text.lower()
    score = 0.0
    matches = 0

    for word, weight in POSITIVE_WORDS.items():
        count = text_lower.count(word)
        if count > 0:
            score += weight * min(count, 3)
            matches += count

    for word, weight in NEGATIVE_WORDS.items():
        count = text_lower.count(word)
        if count > 0:
            score += weight * min(count, 3)
            matches += count

    # Normaliser entre -1 et +1
    if matches == 0:
        return 0.0
    normalized = score / (matches ** 0.5 + 1)
    return max(-1.0, min(1.0, normalized))


def detect_event(text):
    """Detecte le type d'evenement."""
    for event, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return event
    return "autre"


def impact_from_score(score):
    if score > 0.2:
        return "haussier"
    elif score < -0.2:
        return "baissier"
    return "neutre"


# =========================================================
# 2) LLAMA (uniquement pour les news fortes)
# =========================================================
MODEL_LLAMA = "llama-3.1-8b-instant"  # modele leger = moins de tokens
SLEEP_LLAMA = 4.5  # secondes (13 req/min, sous la limite)


def llama_sentiment(client, title, summary=""):
    """Sentiment par Llama (modele leger)."""
    text = f"Titre : {title}\nResume : {summary[:200]}"
    prompt = """Analyse cette news bancaire. Reponds en JSON :
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
# 3) PIPELINE
# =========================================================
def score_news():
    path = f"{DATA_RAW}/banks_news.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[sentiment] {len(df)} news a analyser")

    # -------- Etape 1 : analyse par mots-cles (instantanee) --------
    print("\n[sentiment] === ETAPE 1 : Mots-cles ===")
    kw_sentiments = []
    events = []

    for _, row in df.iterrows():
        text = (row["title"] or "") + " " + (row.get("summary", "") or "")
        kw_sentiments.append(keyword_sentiment(text))
        events.append(detect_event(text))

    df["sentiment_kw"] = kw_sentiments
    df["event_type"] = events
    df["impact"] = df["sentiment_kw"].apply(impact_from_score)

    print(f"  Positifs (kw > +0.2) : {(df['sentiment_kw'] > 0.2).sum()}")
    print(f"  Negatifs (kw < -0.2) : {(df['sentiment_kw'] < -0.2).sum()}")
    print(f"  Neutres              : {((df['sentiment_kw'] >= -0.2) & (df['sentiment_kw'] <= 0.2)).sum()}")

    # -------- Etape 2 : Llama sur les news fortes --------
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

        # Fusionner
        df = df.merge(
            strong_df[["title", "sentiment_llama", "raison"]],
            on="title", how="left"
        )
    else:
        df["sentiment_llama"] = None
        df["raison"] = ""

    # -------- Sentiment final : moyenne ponderee --------
    def final_score(row):
        kw = row["sentiment_kw"]
        ll = row.get("sentiment_llama")
        if pd.notna(ll):
            return 0.4 * kw + 0.6 * ll
        return kw

    df["sentiment"] = df.apply(final_score, axis=1)
    df["impact"] = df["sentiment"].apply(impact_from_score)

    # -------- Sauvegarde --------
    out = f"{DATA_RAW}/banks_news_scored.parquet"
    df.to_parquet(out)
    print(f"\n[sentiment] Sauvegarde -> {out}")

    # -------- Stats --------
    print("\n" + "="*50)
    print("STATISTIQUES FINALES")
    print("="*50)
    print(f"Sentiment moyen : {df['sentiment'].mean():+.3f}")
    print(f"  Positifs : {(df['sentiment'] > 0.2).sum()}")
    print(f"  Negatifs : {(df['sentiment'] < -0.2).sum()}")
    print(f"  Neutres  : {((df['sentiment'] >= -0.2) & (df['sentiment'] <= 0.2)).sum()}")

    print("\nPar banque :")
    print(df.groupby("tickers")["sentiment"].agg(["mean", "count"]).round(3))

    print("\nTypes d'evenements :")
    print(df["event_type"].value_counts())

    return df


if __name__ == "__main__":
    score_news()