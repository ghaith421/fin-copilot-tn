import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import json
import time
from groq import Groq
from config import DATA_RAW, GROQ_API_KEY

MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 10
SLEEP = 4
MAX_PER_RUN = 60  # Limite pour eviter le rate limit


def llama_batch(client, news_list):
    items = ""
    for i, (title, summary) in enumerate(news_list):
        items += f"\n--- NEWS {i+1} ---\nTitre: {title}\nResume: {summary[:120]}\n"

    prompt = f"""Analyse ces {len(news_list)} news bancaires tunisiennes.

Pour chaque news, donne un sentiment entre -1 (tres negatif) et +1 (tres positif).
- Neutre/banale → 0.0
- Positive → +0.1 a +1
- Negative → -1 a -0.1

Reponds en JSON :
{{"results": [
  {{"id": 1, "sentiment": 0.5, "raison": "..."}},
  ...
]}}

{items}"""

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un analyste financier. Reponds en JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        data = json.loads(r.choices[0].message.content)
        results = sorted(data.get("results", []), key=lambda x: x.get("id", 0))
        return [
            {"sentiment": float(r.get("sentiment", 0)), "raison": r.get("raison", "")[:100]}
            for r in results
        ]
    except Exception as ex:
        print(f"    [erreur] {str(ex)[:80]}")
        return [{"sentiment": 0.0, "raison": ""} for _ in news_list]


def enhance_with_llama():
    """Enrichit les news avec sentiment = 0 en utilisant Llama."""
    path = f"{DATA_RAW}/banks_news_scored.parquet"
    df = pd.read_parquet(path).reset_index(drop=True)

    mask_zero = df["sentiment"] == 0
    n_zero = mask_zero.sum()

    if n_zero == 0:
        print("[auto] Toutes les news ont un sentiment")
        return df

    if not GROQ_API_KEY:
        print("[auto] Pas de cle Groq")
        return df

    to_process = df[mask_zero].head(MAX_PER_RUN).copy().reset_index(drop=True)
    print(f"[auto] {n_zero} news sans sentiment, on en traite {len(to_process)} ce run")

    client = Groq(api_key=GROQ_API_KEY)

    n_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    all_scores = []
    all_raisons = []

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(to_process))
        batch_df = to_process.iloc[start:end]

        news_list = [(row["title"], row.get("summary", "")) for _, row in batch_df.iterrows()]
        results = llama_batch(client, news_list)

        while len(results) < len(news_list):
            results.append({"sentiment": 0.0, "raison": ""})

        for r in results[:len(news_list)]:
            all_scores.append(r["sentiment"])
            all_raisons.append(r["raison"])

        print(f"    Batch {b+1}/{n_batches}")
        time.sleep(SLEEP)

    to_process["sentiment_llama"] = all_scores
    to_process["raison"] = all_raisons

    df = df.merge(
        to_process[["title", "sentiment_llama", "raison"]],
        on="title",
        how="left",
        suffixes=("", "_new"),
    )

    mask_apply = (df["sentiment"] == 0) & df["sentiment_llama"].notna()
    df.loc[mask_apply, "sentiment"] = df.loc[mask_apply, "sentiment_llama"]

    df["impact"] = df["sentiment"].apply(
        lambda s: "haussier" if s > 0.2 else ("baissier" if s < -0.2 else "neutre")
    )

    for col in ["sentiment_llama_new", "raison_new"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.to_parquet(path)

    n_nonzero = (df["sentiment"] != 0).sum()
    print(f"[auto] Sauvegarde -> {path}")
    print(f"[auto] News avec sentiment != 0 : {n_nonzero} / {len(df)} ({100*n_nonzero/len(df):.1f}%)")
    return df


if __name__ == "__main__":
    enhance_with_llama()