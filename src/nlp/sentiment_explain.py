import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import re
import pandas as pd
from config import DATA_RAW


def explain_sentiment(text):
    """
    Retourne une explication detaillee du sentiment.
    Montre quels mots ont contribue (positif ou negatif).
    """
    from src.nlp.sentiment_fr import (
        POSITIVE_STEMS, NEGATIVE_STEMS, POSITIVE_EN, NEGATIVE_EN,
        POSITIVE_BIGRAMS, NEGATIVE_BIGRAMS, POSITIVE_TRIGRAMS, NEGATIVE_TRIGRAMS,
        INTENSIFIERS, NEGATIONS, BLACKLIST, match_stem,
    )

    text = (text or "").strip()
    text_lower = text.lower()
    words = re.findall(r"\b[\w']+\b", text_lower)

    contributions = []
    score = 0.0

    # Trigrammes
    for tg, weight in POSITIVE_TRIGRAMS.items():
        if tg in text_lower:
            contributions.append({"word": tg, "weight": weight, "type": "trigram_positif"})
            score += weight
    for tg, weight in NEGATIVE_TRIGRAMS.items():
        if tg in text_lower:
            contributions.append({"word": tg, "weight": weight, "type": "trigram_negatif"})
            score += weight

    # Bigrammes
    for bg, weight in POSITIVE_BIGRAMS.items():
        count = text_lower.count(bg)
        if count > 0:
            contributions.append({"word": bg, "weight": weight, "type": "bigram_positif"})
            score += weight
    for bg, weight in NEGATIVE_BIGRAMS.items():
        count = text_lower.count(bg)
        if count > 0:
            contributions.append({"word": bg, "weight": weight, "type": "bigram_negatif"})
            score += weight

    # Mots individuels (avec deduplication)
    seen_words = set()
    for i, word in enumerate(words):
        if word in BLACKLIST:
            continue
        if word in seen_words:
            continue

        weight = match_stem(word, POSITIVE_STEMS)
        wtype = "mot_positif"
        if weight == 0.0:
            weight = match_stem(word, NEGATIVE_STEMS)
            wtype = "mot_negatif"

        # Anglais : mot exact
        if weight == 0.0 and word in POSITIVE_EN:
            weight = POSITIVE_EN[word]
            wtype = "mot_en_positif"
        if weight == 0.0 and word in NEGATIVE_EN:
            weight = NEGATIVE_EN[word]
            wtype = "mot_en_negatif"

        if weight == 0.0:
            continue

        # Negation
        context = words[max(0, i - 4):i]
        has_neg = any(w in NEGATIONS for w in context)
        if has_neg:
            weight = -weight
            wtype += "_inverse"

        # Intensificateur
        intens_context = words[max(0, i - 2):i]
        intens_found = None
        for w in intens_context:
            if w in INTENSIFIERS:
                weight *= INTENSIFIERS[w]
                intens_found = w
                break

        contributions.append({
            "word": word,
            "weight": weight,
            "type": wtype,
            "intensifier": intens_found,
        })
        score += weight
        seen_words.add(word)

    # Normalisation (IDENTIQUE a sentiment_fr.py)
    if len(contributions) == 0:
        return {
            "text": text,
            "score": 0.0,
            "contributions": [],
            "n_signals": 0,
            "summary": "Aucun signal detecte (news neutre)",
        }

    normalized = score / (len(contributions) + 1)
    normalized = max(-1.0, min(1.0, normalized))

    contributions = sorted(contributions, key=lambda c: abs(c["weight"]), reverse=True)

    return {
        "text": text,
        "score": round(normalized, 3),
        "contributions": contributions,
        "n_signals": len(contributions),
        "summary": _generate_summary(normalized, contributions),
    }


def _generate_summary(score, contributions):
    """Genere un resume en langage naturel."""
    if abs(score) < 0.15:
        return f"News globalement neutre (score {score:+.2f})"

    top = contributions[:3]
    pos_words = [c["word"] for c in top if c["weight"] > 0]
    neg_words = [c["word"] for c in top if c["weight"] < 0]

    if score > 0.15:
        summary = f"News positive (score {score:+.2f})"
        if pos_words:
            summary += f" - mots cles : {', '.join(pos_words[:3])}"
        return summary
    else:
        summary = f"News negative (score {score:+.2f})"
        if neg_words:
            summary += f" - mots cles : {', '.join(neg_words[:3])}"
        return summary


def explain_news(title, summary=""):
    text = (title or "") + ". " + (summary or "")
    return explain_sentiment(text)


if __name__ == "__main__":
    tests = [
        "ZB Tunisie : Banque Zitouna et l'OIT s'allient pour doter les ingenieurs d'une offre de financement",
        "BIAT : Une transaction de bloc realisee pour 10,5 MD",
        "ATB : resultats en baisse, deficit de 20 MTND",
        "UBCI : solidite financiere confirmee, croissance soutenue",
        "BH Bank : pas de benefice cette annee",
    ]

    for t in tests:
        print(f"\n{'='*70}")
        print(f"NEWS : {t}")
        print(f"{'='*70}")
        result = explain_sentiment(t)
        print(f"Score : {result['score']:+.3f}")
        print(f"Resume : {result['summary']}")
        print(f"\nContributions :")
        for c in result["contributions"][:5]:
            emoji = "🟢" if c["weight"] > 0 else "🔴"
            intens = f" (x{c['intensifier']})" if c.get("intensifier") else ""
            print(f"  {emoji} '{c['word']}' -> {c['weight']:+.2f} [{c['type']}]{intens}")