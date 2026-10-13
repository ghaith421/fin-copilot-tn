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
            contributions.append({"word": bg, "weight": weight * min(count, 2), "type": "bigram_positif"})
            score += weight * min(count, 2)
    for bg, weight in NEGATIVE_BIGRAMS.items():
        count = text_lower.count(bg)
        if count > 0:
            contributions.append({"word": bg, "weight": weight * min(count, 2), "type": "bigram_negatif"})
            score += weight * min(count, 2)

    # Mots individuels
    for i, word in enumerate(words):
        if word in BLACKLIST:
            continue

        weight = match_stem(word, POSITIVE_STEMS)
        wtype = "mot_positif"
        if weight == 0.0:
            weight = match_stem(word, NEGATIVE_STEMS)
            wtype = "mot_negatif"
        if weight == 0.0 and len(word) >= 4:
            for en_stem, w in POSITIVE_EN.items():
                if en_stem in word or word in en_stem:
                    weight = w
                    wtype = "mot_en_positif"
                    break
        if weight == 0.0 and len(word) >= 4:
            for en_stem, w in NEGATIVE_EN.items():
                if en_stem in word or word in en_stem:
                    weight = w
                    wtype = "mot_en_negatif"
                    break

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

    # Normalisation
    if len(contributions) == 0:
        return {
            "text": text,
            "score": 0.0,
            "contributions": [],
            "summary": "Aucun signal detecte (news neutre)",
        }

    normalized = score / (len(contributions) ** 0.5 + 0.5)
    normalized = max(-1.0, min(1.0, normalized))

    # Trier par contribution absolue
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
    if abs(score) < 0.1:
        return "News globalement neutre"

    # Top 3 contributions
    top = contributions[:3]
    pos_words = [c["word"] for c in top if c["weight"] > 0]
    neg_words = [c["word"] for c in top if c["weight"] < 0]

    if score > 0.1:
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
    """Fonction pratique pour une news complete."""
    text = (title or "") + ". " + (summary or "")
    return explain_sentiment(text)


if __name__ == "__main__":
    tests = [
        "BIAT annonce un PNB en hausse de 15%",
        "STB : amende de 5 millions de dinars suite a une fraude",
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