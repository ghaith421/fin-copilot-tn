import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_embeddings = None
_news_index = None
_impact = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_embeddings():
    """Construit les embeddings de toutes les news (1 fois)."""
    path = f"{DATA_RAW}/banks_news_scored.parquet"
    if not os.path.exists(path):
        print("[twins] Pas de news disponibles")
        return

    news = pd.read_parquet(path).reset_index(drop=True)
    print(f"[twins] {len(news)} news a encoder")

    texts = []
    for _, r in news.iterrows():
        t = (r.get("title", "") or "") + ". " + (r.get("summary", "") or "")
        texts.append(t.strip()[:500])

    model = get_model()
    print("[twins] Encodage en cours...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, batch_size=32)

    np.save(f"{DATA_PROCESSED}/news_embeddings.npy", embeddings)

    idx = news[["date", "tickers", "title", "sentiment"]].copy()
    idx.to_parquet(f"{DATA_PROCESSED}/news_index.parquet")

    print(f"[twins] Embeddings : {embeddings.shape}")
    print(f"[twins] Index : {len(idx)} news")
    return embeddings, idx


def _load_all():
    global _embeddings, _news_index, _impact

    if _embeddings is None:
        path = f"{DATA_PROCESSED}/news_embeddings.npy"
        if not os.path.exists(path):
            print("[twins] Embeddings manquants, lance d'abord: python -m src.analysis.news_twins --build")
            return None, None
        _embeddings = np.load(path)

    if _news_index is None:
        _news_index = pd.read_parquet(f"{DATA_PROCESSED}/news_index.parquet")

    if _impact is None:
        try:
            _impact = pd.read_parquet(f"{DATA_RAW}/news_impact_clean.parquet")
        except Exception:
            _impact = pd.DataFrame()

    return _embeddings, _news_index


def find_twins(query_text, ticker=None, k=5):
    """Trouve les k news passees les plus similaires."""
    from sklearn.metrics.pairwise import cosine_similarity

    embeddings, index = _load_all()
    if embeddings is None:
        return pd.DataFrame()

    model = get_model()
    q_emb = model.encode([query_text], convert_to_numpy=True)

    # Filtrer par ticker si demande
    if ticker:
        mask = (index["tickers"] == ticker).values
        idx_filtered = np.where(mask)[0]
        if len(idx_filtered) == 0:
            idx_filtered = np.arange(len(index))
    else:
        idx_filtered = np.arange(len(index))

    sims = cosine_similarity(q_emb, embeddings[idx_filtered])[0]
    top_local = np.argsort(sims)[::-1][:k * 2]
    top_global = idx_filtered[top_local]

    result = index.iloc[top_global].copy().reset_index(drop=True)
    result["similarity"] = sims[top_local]

    # Deduplication par titre
    result = result.drop_duplicates(subset=["title"], keep="first").head(k).reset_index(drop=True)

    # Ajouter l'impact
    if len(_impact) > 0:
        impact_small = _impact[["title", "var_j1_pct", "var_j5_pct", "var_j20_pct", "verdict"]].drop_duplicates(subset=["title"])
        result = result.merge(impact_small, on="title", how="left")

    return result


def predict_impact_from_twins(query_text, ticker=None, k=5):
    """
    Predit l'impact base sur les jumeaux.
    Retourne un dict avec l'impact moyen + nb de jumeaux utilises.
    """
    twins = find_twins(query_text, ticker=ticker, k=k)
    if len(twins) == 0:
        return None

    valid_j1 = twins["var_j1_pct"].dropna() if "var_j1_pct" in twins.columns else pd.Series()
    valid_j5 = twins["var_j5_pct"].dropna() if "var_j5_pct" in twins.columns else pd.Series()
    valid_j20 = twins["var_j20_pct"].dropna() if "var_j20_pct" in twins.columns else pd.Series()

    return {
        "n_twins": len(twins),
        "j1_mean": valid_j1.mean() if len(valid_j1) > 0 else None,
        "j5_mean": valid_j5.mean() if len(valid_j5) > 0 else None,
        "j20_mean": valid_j20.mean() if len(valid_j20) > 0 else None,
        "j1_hit_rate": (valid_j1 > 0).mean() * 100 if len(valid_j1) > 0 else None,
        "j5_hit_rate": (valid_j5 > 0).mean() * 100 if len(valid_j5) > 0 else None,
        "twins": twins,
    }


if __name__ == "__main__":
    import sys as _sys
    if "--build" in _sys.argv:
        build_embeddings()
    else:
        # Test
        print("=== Test News Twin Finder ===")
        test_text = "BIAT annonce des resultats en hausse"
        result = predict_impact_from_twins(test_text, ticker="BIAT", k=5)
        if result:
            print(f"\nNombre de jumeaux : {result['n_twins']}")
            print(f"Impact moyen J+5  : {result['j5_mean']:.2f}%" if result['j5_mean'] else "Pas d'impact J+5")
            print(f"Impact moyen J+20 : {result['j20_mean']:.2f}%" if result['j20_mean'] else "Pas d'impact J+20")
            print(f"\nJumeaux trouves :")
            for _, t in result["twins"].iterrows():
                sim = t.get("similarity", 0)
                j5 = t.get("var_j5_pct", np.nan)
                j5_str = f"{j5:+.2f}%" if pd.notna(j5) else "N/A"
                print(f"  [{sim:.3f}] {t['title'][:70]} | J+5: {j5_str}")