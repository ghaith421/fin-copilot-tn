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
# DICTIONNAIRES ENRICHIS
# =========================================================
POSITIVE_WORDS = {
    # Croissance
    "hausse": 1.0, "augmentation": 0.8, "croissance": 0.9, "progression": 0.9,
    "amélioration": 0.8, "amelioration": 0.8, "accélération": 0.8, "acceleration": 0.8,
    "expansion": 0.8, "essor": 0.8, "développement": 0.6, "developpement": 0.6,
    "boost": 0.7, "rebond": 0.8, "surperform": 0.9, "surperforme": 0.8,
    # Performance
    "record": 1.0, "excellent": 1.0, "performant": 0.9, "solide": 0.7,
    "robuste": 0.6, "positif": 0.7, "favorable": 0.7, "optimiste": 0.7,
    "bénéfice": 1.0, "benefice": 1.0, "profit": 0.9, "gains": 0.8,
    "rentabilité": 0.9, "rentabilite": 0.9, "rentable": 0.8,
    "dividende": 0.8, "rendement": 0.6,
    # Succès
    "succès": 0.9, "succes": 0.9, "réussi": 0.8, "reussi": 0.8,
    "victoire": 0.8, "gagnant": 0.7, "récompense": 0.7, "recompense": 0.7,
    "meilleur": 0.7, "leader": 0.6, "premier": 0.5,
    # Actions positives
    "investissement": 0.5, "investir": 0.6, "partenariat": 0.6,
    "acquisition": 0.7, "lancement": 0.5, "nouveau": 0.4,
    "collaboration": 0.5, "accord": 0.5, "signature": 0.4,
    "innovation": 0.7, "digitale": 0.4, "modernisation": 0.5,
}

NEGATIVE_WORDS = {
    # Baisse
    "baisse": -1.0, "diminution": -0.8, "chute": -1.0, "effondrement": -1.0,
    "recul": -0.8, "dégradation": -0.9, "degradation": -0.9, "déclin": -0.8,
    "declin": -0.8, "contraction": -0.7, "chute": -1.0,
    # Pertes
    "perte": -1.0, "pertes": -1.0, "déficit": -0.9, "deficit": -0.9,
    "difficulté": -0.8, "difficulte": -0.8, "crise": -0.9,
    "problème": -0.7, "probleme": -0.7, "risque": -0.6,
    "sous-performe": -0.7, "faiblesse": -0.7, "fragile": -0.6,
    "mal": -0.6, "pire": -0.7,
    # Sanctions
    "amende": -0.9, "sanction": -0.9, "pénalité": -0.9, "penalite": -0.9,
    "condamnation": -1.0, "enquête": -0.7, "enquete": -0.7, "fraude": -1.0,
    "scandale": -1.0, "litige": -0.7, "contentieux": -0.7,
    "corruption": -1.0, "blanchiment": -1.0,
    # Actions négatives
    "avertissement": -0.7, "alerte": -0.6, "négatif": -0.7, "negatif": -0.7,
    "défavorable": -0.7, "defavorable": -0.7, "pessimiste": -0.7,
    "inquiétude": -0.7, "inquietude": -0.7,
    "fermeture": -0.7, "licenciement": -0.8, "restructuration": -0.5,
    "grève": -0.7, "greve": -0.7, "blocage": -0.7,
    "démission": -0.6, "demission": -0.6,
    "suspension": -0.7, "retrait": -0.6,
}

# =========================================================
# BIGRAMMES (expressions de 2 mots) - plus fiables
# =========================================================
POSITIVE_BIGRAMS = {
    "en hausse": 1.2, "en progression": 1.1, "en croissance": 1.2,
    "en augmentation": 1.0, "en amélioration": 1.0,
    "bon résultat": 1.1, "bons résultats": 1.2, "bons resultats": 1.2,
    "bénéfice net": 1.0, "resultat net": 1.0, "résultat net": 1.0,
    "chiffre d'affaires": 0.6,
    "prise de participation": 0.7, "levée de fonds": 0.8, "levee de fonds": 0.8,
    "plan de développement": 0.7, "plan de developpement": 0.7,
    "hausse du cours": 1.0, "hausse des bénéfices": 1.2, "hausse des benefices": 1.2,
    "nouveau partenariat": 0.7, "nouvelle acquisition": 0.8,
    "nouveau directeur": 0.5, "nouveau produit": 0.6,
    "attention positive": 0.6, "perspective favorable": 0.8,
    "amélioration significative": 1.0, "croissance soutenue": 1.1,
}

NEGATIVE_BIGRAMS = {
    "en baisse": -1.2, "en recul": -1.0, "en chute": -1.2,
    "en difficulté": -1.0, "en difficulte": -1.0, "en crise": -1.1,
    "perte nette": -1.1, "resultat négatif": -1.0, "résultat négatif": -1.0,
    "baisse du cours": -1.0, "baisse des bénéfices": -1.2, "baisse des benefices": -1.2,
    "chute des crédits": -1.0, "chute des credits": -1.0,
    "dégradation de la qualité": -1.0, "degradation de la qualite": -1.0,
    "créances douteuses": -0.9, "creances douteuses": -0.9,
    "non-performing": -0.9, "npl": -0.8, "impayés": -0.9, "impayes": -0.9,
    "amende de": -1.0, "sanction de": -0.9,
    "mise en cause": -0.7, "mise en examen": -0.9,
    "démission du": -0.7, "demission du": -0.7,
    "suspension de": -0.8, "retrait de": -0.7,
    "avertissement de": -0.8, "alerte de": -0.7,
    "faible performance": -0.8, "mauvaise performance": -0.9,
    "perspective négative": -0.8, "perspective negative": -0.8,
}

# =========================================================
# MOTS CONTEXTUELS
# =========================================================
# Ces mots n'ont pas de sentiment propre, mais renforcent un contexte
INTENSIFIERS = {
    "fortement": 1.3, "significativement": 1.3, "considerablement": 1.3,
    "tres": 1.2, "très": 1.2, "extrêmement": 1.4, "extremement": 1.4,
    "notablement": 1.2, "nettement": 1.2, "largement": 1.2,
    "faiblement": 0.7, "legèrement": 0.7, "legerement": 0.7, "peu": 0.6,
}

NEGATIONS = ["pas", "non", "aucun", "aucune", "sans", "jamais", "ni",
             "ne", "n'", "point", "guère", "guere", "rien"]

# Mots neutres (indiquent une news generale)
NEUTRAL_KEYWORDS = [
    "assemblee", "assemblée", "ag", "ago", "age",
    "salon", "exposition", "conference", "conférence", "forum",
    "interview", "communique", "communiqué", "declaration", "déclaration",
    "reunion", "réunion", "seance", "séance", "visite",
]

EVENT_PATTERNS = {
    "resultats": r"(résultat|resultat|résultats|resultats|bénéfice|benefice|pnb|bilan|chiffre)",
    "dividende": r"(dividende|distribution|assemblee generale|assemblée générale|AGO)",
    "regulation": r"(BCT|banque centrale|régulation|regulation|loi|décret|decret|amende|sanction)",
    "macro": r"(inflation|taux|BCE|FED|économie|economie|croissance du PIB|chômage|chomage)",
    "fusion": r"(fusion|acquisition|OPA|partenariat|rachat)",
    "direction": r"(directeur|PDG|DG|président|president|nomination|démission|demission)",
    "npl": r"(creance|créance|impaye|impayé|npl|non.performing|douteux)",
}

SOURCE_WEIGHTS = {
    "reuters": 1.5, "bloomberg": 1.5, "afp": 1.5,
    "ilboursa": 1.2, "tustex": 1.2, "lapresse": 1.2, "businessnews": 1.2,
    "yahoo": 1.0, "cnbc": 1.0, "marketwatch": 1.0,
    "google_news": 1.0,
}

MODEL_LLAMA = "openai/gpt-oss-20b"
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 6


# =========================================================
# SCORING ENRICHI
# =========================================================
def extract_numbers_sentiment(text):
    """Detecte les nombres avec +/- pour un signal."""
    score = 0.0
    text_lower = text.lower()

    # Chercher +X% ou -X%
    # Ex: "+5.2%", "+12 %", "-3.4%"
    plus_matches = re.findall(r"\+\s*(\d+[.,]?\d*)\s*%", text_lower)
    minus_matches = re.findall(r"-\s*(\d+[.,]?\d*)\s*%", text_lower)

    for m in plus_matches:
        v = float(m.replace(",", "."))
        score += min(v / 20, 1.0)  # +20% = +1.0
    for m in minus_matches:
        v = float(m.replace(",", "."))
        score -= min(v / 20, 1.0)

    # Mots "hausse de X%" / "baisse de X%"
    hausse_match = re.findall(r"hausse\s+de\s+(\d+[.,]?\d*)\s*%", text_lower)
    baisse_match = re.findall(r"baisse\s+de\s+(\d+[.,]?\d*)\s*%", text_lower)

    for m in hausse_match:
        v = float(m.replace(",", "."))
        score += min(v / 15, 1.0)
    for m in baisse_match:
        v = float(m.replace(",", "."))
        score -= min(v / 15, 1.0)

    return score


def keyword_sentiment_enriched(text):
    """
    Scoring enrichi :
    1. Mots simples
    2. Bigrammes
    3. Negations
    4. Intensificateurs
    5. Nombres +/-
    """
    text = (text or "").strip()
    text_lower = text.lower()
    words = re.findall(r"\b[\w']+\b", text_lower)

    score = 0.0
    matches = 0
    signals = []

    # ---- 1. BIGRAMMES (priorite haute) ----
    for bg, weight in POSITIVE_BIGRAMS.items():
        count = text_lower.count(bg)
        if count > 0:
            score += weight * min(count, 2)
            matches += count
            signals.append(f"+bg:{bg}")

    for bg, weight in NEGATIVE_BIGRAMS.items():
        count = text_lower.count(bg)
        if count > 0:
            score += weight * min(count, 2)
            matches += count
            signals.append(f"-bg:{bg}")

    # ---- 2. MOTS SIMPLES avec negation + intensificateur ----
    for i, word in enumerate(words):
        weight = 0.0
        if word in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[word]
        elif word in NEGATIVE_WORDS:
            weight = NEGATIVE_WORDS[word]
        else:
            continue

        # Verifier negation (4 mots avant)
        context = words[max(0, i - 4):i]
        has_neg = any(n in " ".join(context) for n in NEGATIONS)
        if has_neg:
            weight = -weight

        # Verifier intensificateur (2 mots avant)
        intens_context = words[max(0, i - 2):i]
        for w in intens_context:
            if w in INTENSIFIERS:
                weight *= INTENSIFIERS[w]
                break

        score += weight
        matches += 1

    # ---- 3. NOMBRES +/- ----
    num_score = extract_numbers_sentiment(text)
    if abs(num_score) > 0.01:
        score += num_score
        matches += 1

    # ---- 4. NEUTRALITE (assemblee, salon, etc.) ----
    is_neutral = any(kw in text_lower for kw in NEUTRAL_KEYWORDS)
    if is_neutral and matches == 0:
        return 0.0, 0, ["neutral"]

    # ---- Normalisation ----
    if matches == 0:
        return 0.0, 0, []

    normalized = score / (matches ** 0.5 + 0.5)
    return float(np.clip(normalized, -1.0, 1.0)), matches, signals


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


# =========================================================
# LLAMA PAR BATCH
# =========================================================
def llama_batch(client, news_list):
    items_text = ""
    for i, (title, summary) in enumerate(news_list):
        items_text += f"\n--- NEWS {i+1} ---\nTitre: {title}\nResume: {summary[:150]}\n"

    prompt = f"""Analyse ces {len(news_list)} news bancaires tunisiennes.

Pour CHAQUE news, donne un sentiment entre -1 (tres negatif) et +1 (tres positif).
- Neutre/banale → 0.0
- Positive → entre +0.1 et +1
- Negative → entre -1 et -0.1

Reponds UNIQUEMENT en JSON valide avec ce format :
{{"results": [
  {{"id": 1, "sentiment": 0.5, "raison": "..."}},
  ...
]}}

{items_text}"""

    try:
        r = client.chat.completions.create(
            model=MODEL_LLAMA,
            messages=[
                {"role": "system", "content": "Tu es un analyste financier. Reponds UNIQUEMENT en JSON valide."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        data = json.loads(r.choices[0].message.content)
        results = data.get("results", [])
        results = sorted(results, key=lambda x: x.get("id", 0))
        return [
            {"sentiment": float(r.get("sentiment", 0)), "raison": r.get("raison", "")[:100]}
            for r in results
        ]
    except Exception as ex:
        print(f"    [erreur batch] {str(ex)[:80]}")
        return [{"sentiment": 0.0, "raison": ""} for _ in news_list]


# =========================================================
# PIPELINE
# =========================================================
def score_news_keywords_only():
    """Version rapide : scoring enrichi (mots-cles + bigrammes + contexte)."""
    path = f"{DATA_RAW}/banks_news.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[sentiment-kw] {len(df)} news")

    sentiments = []
    matches_list = []
    events = []
    signals_list = []

    for _, row in df.iterrows():
        text = (row["title"] or "") + ". " + (row.get("summary", "") or "")
        s, m, sig = keyword_sentiment_enriched(text)
        sentiments.append(s)
        matches_list.append(m)
        events.append(detect_event(text))
        signals_list.append(",".join(sig[:3]))

    df["sentiment_kw"] = sentiments
    df["sentiment"] = sentiments
    df["kw_matches"] = matches_list
    df["kw_signals"] = signals_list
    df["event_type"] = events
    df["impact"] = df["sentiment"].apply(impact_from_score)

    out = f"{DATA_RAW}/banks_news_scored.parquet"
    df.to_parquet(out)

    n_nonzero = (df["sentiment"] != 0).sum()
    print(f"[sentiment-kw] Sentiment moyen : {df['sentiment'].mean():+.3f}")
    print(f"[sentiment-kw] News avec sentiment != 0 : {n_nonzero} / {len(df)}")
    print("\nPar banque :")
    print(df.groupby("tickers")["sentiment"].agg(["mean", "count"]).round(3))
    return df


def score_news():
    """Pipeline complet : enrichi + Llama par batch."""
    path = f"{DATA_RAW}/banks_news.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[sentiment] {len(df)} news")

    print("\n[sentiment] === ETAPE 1 : Scoring enrichi (mots-cles + bigrammes) ===")
    sentiments = []
    matches_list = []
    events = []
    for _, row in df.iterrows():
        text = (row["title"] or "") + ". " + (row.get("summary", "") or "")
        s, m, _ = keyword_sentiment_enriched(text)
        sentiments.append(s)
        matches_list.append(m)
        events.append(detect_event(text))

    df["sentiment_kw"] = sentiments
    df["kw_matches"] = matches_list
    df["event_type"] = events
    df["source_weight"] = df["source"].apply(source_weight) if "source" in df.columns else 1.0

    n_nonzero = (df["sentiment_kw"] != 0).sum()
    print(f"  News avec sentiment != 0 : {n_nonzero} / {len(df)}")
    print(f"  News sans signal : {(df['sentiment_kw'] == 0).sum()}")

    # Llama sur les news faibles
    needs_llama_mask = df["sentiment_kw"].abs() < 0.3
    needs_llama = df[needs_llama_mask].copy().reset_index(drop=True)

    if len(needs_llama) > 0 and GROQ_API_KEY:
        print(f"\n[sentiment] === ETAPE 2 : Llama sur {len(needs_llama)} news faibles ===")
        client = Groq(api_key=GROQ_API_KEY)
        all_scores = []
        all_raisons = []

        n_batches = (len(needs_llama) + BATCH_SIZE - 1) // BATCH_SIZE
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(needs_llama))
            batch_df = needs_llama.iloc[start:end]
            news_list = [(row["title"], row.get("summary", "")) for _, row in batch_df.iterrows()]
            results = llama_batch(client, news_list)
            while len(results) < len(news_list):
                results.append({"sentiment": 0.0, "raison": ""})
            for r in results[:len(news_list)]:
                all_scores.append(r["sentiment"])
                all_raisons.append(r["raison"])
            print(f"    Batch {b+1}/{n_batches} : {end}/{len(needs_llama)}")
            time.sleep(SLEEP_BETWEEN_BATCHES)

        needs_llama["sentiment_llama"] = all_scores
        needs_llama["raison"] = all_raisons
        df = df.merge(needs_llama[["title", "sentiment_llama", "raison"]], on="title", how="left")
    else:
        df["sentiment_llama"] = None
        df["raison"] = ""

    def final_score(row):
        kw = row["sentiment_kw"]
        ll = row.get("sentiment_llama")
        if pd.notna(ll) and abs(kw) < 0.3:
            return float(np.clip(0.3 * kw + 0.7 * ll, -1, 1))
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
    print(f"News avec sentiment != 0 : {(df['sentiment'] != 0).sum()} / {len(df)}")
    print("\nPar banque :")
    print(df.groupby("tickers")["sentiment"].agg(["mean", "count"]).round(3))
    return df


if __name__ == "__main__":
    if "--keywords-only" in sys.argv:
        score_news_keywords_only()
    else:
        score_news()