import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import re
import numpy as np
from config import DATA_RAW

# =========================================================
# RACINES POSITIVES
# =========================================================
POSITIVE_STEMS = {
    "hauss": 1.0, "augment": 0.8, "croiss": 0.9, "progress": 0.9,
    "amélior": 0.8, "amelior": 0.8, "accélé": 0.8, "accele": 0.8,
    "expans": 0.8, "essor": 0.8, "développ": 0.6, "developp": 0.6,
    "boost": 0.7, "rebond": 0.8, "surperform": 0.9,
    "record": 1.0, "excellent": 1.0, "performant": 0.9, "solide": 0.7,
    "robust": 0.6, "positif": 0.7, "favorab": 0.7, "optimis": 0.7,
    "bénéfic": 1.0, "benefic": 1.0, "profit": 0.9, "gain": 0.8,
    "rentab": 0.9, "dividend": 0.8, "rendement": 0.6,
    "succès": 0.9, "succes": 0.9, "réussi": 0.8, "reussi": 0.8,
    "victoire": 0.8, "gagn": 0.7, "récompens": 0.7, "recompens": 0.7,
    "meilleur": 0.7, "leader": 0.6,
    "investi": 0.5, "partenariat": 0.6, "acquisition": 0.7,
    "lancement": 0.5, "innov": 0.7, "modernis": 0.5,
    "levée": 0.6, "levee": 0.6, "distribution": 0.5,
    "signature": 0.4, "conclu": 0.5, "obtien": 0.6,
    "décroch": 0.7, "decroch": 0.7, "remport": 0.6,
    "attribu": 0.5, "accordé": 0.5, "accorde": 0.5,
    "solvab": 0.7, "capitalis": 0.6, "upgrad": 0.8,
    "recouvr": 0.5, "assainissement": 0.6,
    "annonce": 0.3, "publi": 0.3, "présent": 0.4, "present": 0.4,
    "réalis": 0.5, "realis": 0.5, "attei": 0.6, "dépass": 0.7, "depass": 0.7,
    "confirme": 0.4, "réaffirme": 0.4, "reaffirme": 0.4,
    "renforce": 0.5, "consolide": 0.5, "développe": 0.5, "developpe": 0.5,
    "augmente": 0.8, "progresse": 0.8, "grimpe": 0.9,
    "opportun": 0.6, "potentiel": 0.4, "prometteur": 0.7,
    "avantage": 0.5, "compétiti": 0.5, "competiti": 0.5,
    "excellence": 0.8,
}

# =========================================================
# RACINES NEGATIVES
# =========================================================
NEGATIVE_STEMS = {
    "baiss": -1.0, "diminu": -0.8, "chut": -1.0, "effondr": -1.0,
    "recul": -0.8, "dégrad": -0.9, "degrad": -0.9, "déclin": -0.8,
    "declin": -0.8, "contract": -0.7,
    "perd": -1.0, "perte": -1.0, "déficit": -0.9, "deficit": -0.9,
    "difficult": -0.8, "crise": -0.9, "problèm": -0.7, "problem": -0.7,
    "risqu": -0.2, "faibl": -0.7, "fragil": -0.6, "pire": -0.7,
    "amend": -0.9, "sanction": -0.9, "pénal": -0.9, "penal": -0.9,
    "condamn": -1.0, "enquêt": -0.7, "enquet": -0.7, "fraud": -1.0,
    "scandal": -1.0, "litig": -0.7, "contentieu": -0.7,
    "corrupt": -1.0, "blanch": -1.0, "suspect": -0.8,
    "averti": -0.7, "alerte": -0.6, "négat": -0.7, "negat": -0.7,
    "défavorab": -0.7, "defavorab": -0.7, "pessimis": -0.7,
    "inquiét": -0.7, "inquiet": -0.7, "fermetur": -0.7,
    "licenci": -0.8, "restructur": -0.5, "grèv": -0.7, "grev": -0.7,
    "blocag": -0.7, "démiss": -0.6, "demiss": -0.6,
    "suspen": -0.7, "retrait": -0.6, "impay": -0.9,
    "douteu": -0.8, "provision": -0.5, "dépréci": -0.7, "depreci": -0.7,
    "défaut": -1.0, "defaut": -1.0, "insolvab": -1.0,
    "illiquid": -0.8, "sous-capital": -0.8,
    "downgrad": -0.9, "déclass": -0.8, "declass": -0.8,
    "non-perform": -0.9, "recouvrement": -0.4,
    "contraint": -0.5, "limitat": -0.5, "restrict": -0.5,
    "désengage": -0.6, "desengage": -0.6, "cessation": -0.6, "cesser": -0.6,
    "manqu": -0.7, "rate": -0.6, "affaibli": -0.7,
    "préoccup": -0.7, "preoccup": -0.7, "menac": -0.7,
    "annul": -0.8, "abandonn": -0.7,
    "poursu": -0.6, "accus": -0.8,
    "fragilis": -0.6,
}

POSITIVE_EN = {
    "growth": 0.9, "increase": 0.8, "rise": 0.9, "gain": 0.8,
    "profit": 0.9, "excellent": 1.0, "strong": 0.8,
    "improve": 0.7, "boost": 0.7, "beat": 0.8, "surge": 0.9,
    "outperform": 0.9, "dividend": 0.8, "upside": 0.7,
    "win": 0.7, "award": 0.6, "leader": 0.6, "best": 0.7,
    "outstanding": 0.8, "exceed": 0.7, "succeed": 0.7,
}

NEGATIVE_EN = {
    "loss": -1.0, "losses": -1.0, "decline": -0.9, "fall": -0.9,
    "drop": -0.9, "decrease": -0.8, "weak": -0.7, "miss": -0.7,
    "concern": -0.6, "risk": -0.2, "warning": -0.7, "default": -1.0,
    "downgrade": -0.9, "plunge": -1.0, "slump": -0.9,
    "underperform": -0.8, "bankrupt": -1.0,
    "penalty": -0.9, "sanction": -0.9, "lawsuit": -0.8,
    "fraud": -1.0, "scandal": -1.0, "suspend": -0.7,
    "problem": -0.7, "fail": -0.8,
}

# =========================================================
# BIGRAMMES
# =========================================================
POSITIVE_BIGRAMS = {
    "en hausse": 1.2, "en progression": 1.1, "en croissance": 1.2,
    "en augmentation": 1.0, "en amélioration": 1.0, "en amelioration": 1.0,
    "bon résultat": 1.1, "bons résultats": 1.2, "bons resultats": 1.2,
    "bénéfice net": 1.0, "résultat net": 1.0, "resultat net": 1.0,
    "chiffre d'affaires": 0.6,
    "prise de participation": 0.7, "levée de fonds": 0.8, "levee de fonds": 0.8,
    "plan de développement": 0.7, "plan de developpement": 0.7,
    "hausse du cours": 1.0, "nouveau partenariat": 0.7,
    "nouvelle acquisition": 0.8, "croissance soutenue": 1.1,
    "hausse du pnb": 1.2, "résultat en hausse": 1.3, "resultat en hausse": 1.3,
    "benefice en hausse": 1.3, "bénéfice en hausse": 1.3,
    "prise de contrôle": 0.6, "signature d'accord": 0.7,
    "amélioration de la": 0.8, "amelioration de la": 0.8,
    "réduction des coûts": 0.6, "reduction des couts": 0.6,
    "partage des risques": 0.4, "gestion des risques": 0.3,
    "maitrise des risques": 0.5, "maîtrise des risques": 0.5,
    "porte à 70": 0.3, "porte a 70": 0.3,
    "porte à 100": 0.3, "porte a 100": 0.3,
    "signe un partenariat": 0.6, "signe un accord": 0.6,
    "hausse des dépôts": 0.8, "hausse des depots": 0.8,
    "croissance du pnb": 1.2, "amélioration du résultat": 1.0,
    "amélioration du resultat": 1.0,
    "résultats solides": 1.1, "resultats solides": 1.1,
    "performance solide": 1.0, "hausse de capital": 0.6,
    "signature de convention": 0.5, "attribution de marché": 0.6,
    "attribution de marche": 0.6,
    "partenariat stratégique": 0.8, "partenariat strategique": 0.8,
    "notation maintenue": 0.5, "perspectives positives": 0.8,
    "nouvelle agence": 0.4, "ouverture de capital": 0.5,
    "nouveau produit": 0.6, "nouveau service": 0.5,
    "bien positionné": 0.5, "bien positionne": 0.5,
    "position solide": 0.7, "solidité financière": 0.8,
    "solidite financiere": 0.8,
    "bénéfice en progression": 1.2, "benefice en progression": 1.2,
}

NEGATIVE_BIGRAMS = {
    "en baisse": -1.2, "en recul": -1.0, "en chute": -1.2,
    "en difficulté": -1.0, "en difficulte": -1.0, "en crise": -1.1,
    "perte nette": -1.1, "résultat négatif": -1.0, "resultat negatif": -1.0,
    "baisse du cours": -1.0, "baisse des bénéfices": -1.2,
    "créances douteuses": -0.9, "creances douteuses": -0.9,
    "loan losses": -1.0, "loan loss": -0.9,
    "non-performing": -0.9, "non performing": -0.9,
    "amende de": -1.0, "sanction de": -0.9,
    "mise en cause": -0.7, "mise en examen": -0.9,
    "suspension de": -0.8, "faible performance": -0.8,
    "dégradation de": -0.9, "degradation de": -0.9,
    "chute des crédits": -1.0, "chute des credits": -1.0,
    "baisse du pnb": -1.2, "résultat en baisse": -1.3, "resultat en baisse": -1.3,
    "perte de confiance": -0.8, "retrait de licence": -1.0,
    "mise sous tutelle": -1.0, "procédure collective": -1.0,
    "problème de liquidité": -0.9, "probleme de liquidite": -0.9,
    "baisse des dépôts": -0.8, "baisse des depots": -0.8,
    "chute du pnb": -1.2, "dégradation du résultat": -1.0,
    "degradation du resultat": -1.0,
    "perte de marché": -0.8, "perte de marche": -0.8,
    "avertissement de la bct": -1.0,
    "mise en demeure": -0.9, "défaut de paiement": -1.0,
    "defaut de paiement": -1.0, "cessation de paiement": -1.0,
    "faible rentabilité": -0.8, "faible rentabilite": -0.8,
    "chute des bénéfices": -1.2, "chute des benefices": -1.2,
    "situation critique": -0.9, "situation préoccupante": -0.8,
    "situation preoccupante": -0.8,
    "baisse des résultats": -1.0, "baisse des resultats": -1.0,
    "perte de clientèle": -0.7, "perte de clientele": -0.7,
}

POSITIVE_TRIGRAMS = {
    "résultat en hausse": 1.3, "resultat en hausse": 1.3,
    "bénéfice en hausse": 1.3, "benefice en hausse": 1.3,
    "hausse du produit net": 1.3,
    "hausse de 10": 0.8, "hausse de 20": 0.9,
    "hausse de 30": 1.0, "hausse de 50": 1.1,
    "croissance du produit net": 1.3,
}

NEGATIVE_TRIGRAMS = {
    "résultat en baisse": -1.3, "resultat en baisse": -1.3,
    "baisse de 10": -0.8, "baisse de 20": -0.9,
    "baisse de 30": -1.0, "baisse de 50": -1.1,
    "perte de confiance": -0.8,
    "chute du résultat": -1.3, "chute du resultat": -1.3,
    "défaut de paiement": -1.0, "defaut de paiement": -1.0,
}

INTENSIFIERS = {
    "fortement": 1.3, "significativement": 1.3, "considérablement": 1.3,
    "considerablement": 1.3, "très": 1.2, "tres": 1.2,
    "extrêmement": 1.4, "extremement": 1.4, "notablement": 1.2,
    "nettement": 1.2, "largement": 1.2, "faiblement": 0.7,
    "légèrement": 0.7, "legerement": 0.7,
    "totalement": 1.3, "entièrement": 1.2,
}

NEGATIONS = {"pas", "non", "aucun", "aucune", "sans", "jamais",
             "ni", "ne", "n'", "point", "guère", "guere", "rien"}

BLACKLIST = {
    "amen", "biat", "atb", "uib", "bna", "bh", "stb", "bt", "bte",
    "ubci", "wifak", "zitouna", "attijari", "tunindex", "tunisie",
    "tunis", "banque", "bank", "bancaire", "bvmt", "bourse",
    "action", "actions", "marche", "marches", "cours",
    "cette", "leurs", "notre", "votre",
}

# Poids maximum par mot (pour eviter le surpoids)
MAX_REPETITIONS = 1  # chaque mot compte UNE SEULE FOIS


def match_stem(word, stem_dict):
    for stem, weight in stem_dict.items():
        if len(stem) >= 4 and stem in word:
            return weight
    return 0.0


def keyword_sentiment_enriched(text):
    """
    Scoring enrichi avec :
    - Trigrammes, bigrammes
    - Mots avec racines
    - Negation
    - Limitation des repetitions (chaque mot compte 1 fois)
    """
    text = (text or "").strip()
    text_lower = text.lower()
    words = re.findall(r"\b[\w']+\b", text_lower)

    score = 0.0
    matches = 0
    seen_stems = set()  # pour eviter les repetitions

    # Trigrammes
    for tg, weight in POSITIVE_TRIGRAMS.items():
        if tg in text_lower:
            score += weight
            matches += 1
    for tg, weight in NEGATIVE_TRIGRAMS.items():
        if tg in text_lower:
            score += weight
            matches += 1

    # Bigrammes
    for bg, weight in POSITIVE_BIGRAMS.items():
        if bg in text_lower:
            score += weight
            matches += 1
    for bg, weight in NEGATIVE_BIGRAMS.items():
        if bg in text_lower:
            score += weight
            matches += 1

    # Mots individuels (avec deduplication par mot)
    seen_words = set()
    for i, word in enumerate(words):
        if word in BLACKLIST:
            continue
        if word in seen_words:
            continue  # deja compte

        weight = match_stem(word, POSITIVE_STEMS)
        if weight == 0.0:
            weight = match_stem(word, NEGATIVE_STEMS)
        if weight == 0.0 and len(word) >= 4:
            for en_stem, w in POSITIVE_EN.items():
                if en_stem in word or word in en_stem:
                    weight = w
                    break
        if weight == 0.0 and len(word) >= 4:
            for en_stem, w in NEGATIVE_EN.items():
                if en_stem in word or word in en_stem:
                    weight = w
                    break

        if weight == 0.0:
            continue

        # Negation
        context = words[max(0, i - 4):i]
        has_neg = any(w in NEGATIONS for w in context)
        if has_neg:
            weight = -weight

        # Intensificateur
        intens_context = words[max(0, i - 2):i]
        for w in intens_context:
            if w in INTENSIFIERS:
                weight *= INTENSIFIERS[w]
                break

        score += weight
        matches += 1
        seen_words.add(word)

    # Nombres signes
    plus_matches = re.findall(r"\+\s*(\d+[.,]?\d*)\s*%", text_lower)
    minus_matches = re.findall(r"-\s*(\d+[.,]?\d*)\s*%", text_lower)
    for m in plus_matches:
        score += min(float(m.replace(",", ".")) / 20, 1.0)
        matches += 1
    for m in minus_matches:
        score -= min(float(m.replace(",", ".")) / 20, 1.0)
        matches += 1

    if matches == 0:
        return 0.0, 0

    # Normalisation avec division plus forte pour eviter les extremes
    normalized = score / (matches + 1)
    return float(np.clip(normalized, -1.0, 1.0)), matches


EVENT_PATTERNS = {
    "resultats": r"(résultat|resultat|résultats|resultats|bénéfice|benefice|pnb|bilan|chiffre)",
    "dividende": r"(dividende|distribution|assemblee generale|assemblée générale|AGO)",
    "regulation": r"(BCT|banque centrale|régulation|regulation|loi|décret|decret|amende|sanction)",
    "macro": r"(inflation|taux|BCE|FED|économie|economie|chômage|chomage)",
    "fusion": r"(fusion|acquisition|OPA|partenariat|rachat)",
    "direction": r"(directeur|PDG|DG|président|president|nomination|démission|demission)",
    "npl": r"(creance|créance|impaye|impayé|npl|non.performing|douteux|loan.loss)",
    "prix": r"(prix|recompense|récompense|distinction|award|laureat|lauréat)",
}


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


def score_news_keywords_only():
    path = f"{DATA_RAW}/banks_news.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)
    print(f"[sentiment] {len(df)} news a analyser")

    sentiments = []
    matches_list = []
    events = []

    for _, row in df.iterrows():
        text = (row["title"] or "") + ". " + (row.get("summary", "") or "")
        s, m = keyword_sentiment_enriched(text)
        sentiments.append(s)
        matches_list.append(m)
        events.append(detect_event(text))

    df["sentiment_kw"] = sentiments
    df["sentiment"] = sentiments
    df["kw_matches"] = matches_list
    df["event_type"] = events
    df["impact"] = df["sentiment"].apply(impact_from_score)

    out = f"{DATA_RAW}/banks_news_scored.parquet"
    df.to_parquet(out)

    n_nonzero = (df["sentiment"] != 0).sum()
    print(f"[sentiment] Sentiment moyen : {df['sentiment'].mean():+.3f}")
    print(f"[sentiment] News avec sentiment != 0 : {n_nonzero} / {len(df)} ({100*n_nonzero/len(df):.1f}%)")
    print("\nPar banque :")
    print(df.groupby("tickers")["sentiment"].agg(["mean", "count"]).round(3))
    return df


if __name__ == "__main__":
    score_news_keywords_only()