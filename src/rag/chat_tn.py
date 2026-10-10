import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from config import DATA_RAW, GROQ_API_KEY

load_dotenv()

MODEL = "qwen/qwen3.8-27b"
BANKS = ["BIAT", "ATB", "AB", "AMEN", "UIB", "BNA", "BH", "STB", "BT", "BTE", "UBCI", "WIFAK"]

BANK_ALIASES = {
    "BIAT":  ["biat", "banque internationale arabe"],
    "ATB":   ["atb", "arab tunisian"],
    "AB":    ["attijari", "ab bank", "attijari bank"],
    "AMEN":  ["amen"],
    "UIB":   ["uib", "union internationale"],
    "BNA":   ["bna", "banque nationale agricole"],
    "BH":    ["bh bank", "banque de l'habitat", "banque habitat"],
    "STB":   ["stb", "societe tunisienne de banque"],
    "BT":    ["banque de tunisie", "bt bank"],
    "BTE":   ["bte", "banque de tunisie et des emirats"],
    "UBCI":  ["ubci", "union bancaire"],
    "WIFAK": ["wifak"],
}


def get_client():
    return Groq(api_key=GROQ_API_KEY)


def detect_bank(question):
    """Detecte la banque mentionnee dans la question."""
    q = question.lower()
    for ticker, aliases in BANK_ALIASES.items():
        if any(a in q for a in aliases):
            return ticker
    return None


# =========================================================
# CONTEXTE
# =========================================================
def build_sector_context():
    """Vue globale du secteur bancaire tunisien."""
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        try:
            news = pd.read_parquet(f"{DATA_RAW}/banks_news.parquet")
        except Exception:
            return "Aucune donnee disponible."

    lines = ["=== SECTEUR BANCAIRE TUNISIEN ==="]
    lines.append(f"Nombre de news analysees : {len(news)}")
    lines.append("")

    stats = news.groupby("tickers").agg(
        sentiment_moyen=("sentiment", "mean") if "sentiment" in news.columns else ("sentiment_kw", "mean"),
        nb_news=("title", "count"),
    ).round(3).sort_values("sentiment_moyen", ascending=False)

    lines.append("Sentiment moyen par banque :")
    for bank, row in stats.iterrows():
        lines.append(f"  - {bank} : {row['sentiment_moyen']:+.3f} ({int(row['nb_news'])} news)")

    return "\n".join(lines)


def build_bank_context(ticker):
    """Contexte detaille pour une banque."""
    try:
        news = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        try:
            news = pd.read_parquet(f"{DATA_RAW}/banks_news.parquet")
        except Exception:
            return f"Aucune news pour {ticker}."

    lines = [f"=== DONNEES POUR {ticker} ==="]
    bnews = news[news["tickers"] == ticker].sort_values("date", ascending=False)

    if len(bnews) == 0:
        return f"Aucune news pour {ticker}."

    sent_col = "sentiment" if "sentiment" in bnews.columns else "sentiment_kw"
    sent_moy = bnews[sent_col].mean()

    lines.append(f"Nombre de news : {len(bnews)}")
    lines.append(f"Sentiment moyen : {sent_moy:+.3f}")

    # Repartition
    pos = (bnews[sent_col] > 0.2).sum()
    neg = (bnews[sent_col] < -0.2).sum()
    lines.append(f"  - Positives : {pos}")
    lines.append(f"  - Negatives : {neg}")
    lines.append(f"  - Neutres : {len(bnews) - pos - neg}")
    lines.append("")

    # Types d'evenements
    if "event_type" in bnews.columns:
        lines.append("Types d'evenements :")
        for evt, cnt in bnews["event_type"].value_counts().items():
            lines.append(f"  - {evt} : {cnt}")
        lines.append("")

    # Top news positives et negatives
    lines.append("Top 5 news positives :")
    for _, r in bnews.nlargest(5, sent_col).iterrows():
        lines.append(f"  - [{r[sent_col]:+.2f}] {r['title'][:120]}")

    lines.append("")
    lines.append("Top 5 news negatives :")
    for _, r in bnews.nsmallest(5, sent_col).iterrows():
        lines.append(f"  - [{r[sent_col]:+.2f}] {r['title'][:120]}")

    # Prix si dispo
    try:
        prices = pd.read_parquet(f"{DATA_RAW}/bvmt_prices.parquet")
        p = prices[prices["ticker"] == ticker].sort_values("date")
        if len(p) > 0:
            last = p.iloc[-1]
            lines.append("")
            lines.append(f"Prix actuel : {last['close']:.2f} DT")
    except Exception:
        pass

    return "\n".join(lines)


# =========================================================
# SYSTEM PROMPT
# =========================================================
SYSTEM_PROMPT = """Tu es Fin Copilot TN, un assistant financier specialise dans le secteur bancaire tunisien.

Tu reponds en francais, de facon claire, pedagogique et concise (150-250 mots max).

## Ce que tu sais faire

1. **Questions sur les banques tunisiennes**
   Utilise les donnees du CONTEXTE (sentiment, news, prix).
   Banques couvertes : BIAT, ATB, AB, AMEN, UIB, BNA, BH, STB, BT, BTE, UBCI, WIFAK.

2. **Questions generales de finance**
   (C'est quoi le PNB ? Comment lire un bilan bancaire ? C'est quoi le CET1 ?)
   Utilise tes connaissances generales.

3. **Analyses comparatives**
   Compare les banques sur leur sentiment, leur actualite.

4. **Interpretation du sentiment**
   Explique ce que signifie un sentiment positif/negatif.

## Regles

- Cite les CHIFFRES exacts du contexte (sentiment, nb news)
- Si une donnee manque, dis-le honnetement
- Ne donne JAMAIS de conseil d'investissement
- Termine TOUJOURS par : "⚠️ Outil pedagogique. Pas un conseil en investissement."
"""


# =========================================================
# FONCTION PRINCIPALE
# =========================================================
def ask(question, history=None):
    """Pose une question au chatbot."""
    client = get_client()

    # Detection de la banque
    bank = detect_bank(question)

    # Contexte
    parts = [build_sector_context()]
    if bank:
        parts.append(build_bank_context(bank))

    context = "\n\n".join(parts)

    user_msg = f"""CONTEXTE ACTUEL :

{context}

---

QUESTION : {question}

Reponds en suivant les regles du system prompt."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    if history:
        messages = [messages[0]] + history + [messages[-1]]

    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )
    return r.choices[0].message.content


# =========================================================
# TEST EN CLI
# =========================================================
if __name__ == "__main__":
    tests = [
        "Bonjour, que peux-tu faire ?",
        "Quelle banque a le meilleur sentiment ?",
        "Analyse BIAT",
        "Compare BIAT et UBCI",
        "C'est quoi le PNB ?",
        "Quelles sont les news negatives sur STB ?",
    ]
    for q in tests:
        print(f"\n{'='*60}\nQ : {q}\n{'='*60}")
        print(ask(q))