import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime
from config import DATA_RAW

st.set_page_config(page_title="Fin Copilot TN", layout="wide", page_icon="🇹🇳")

if "auto_fetch_done" not in st.session_state:
    st.session_state.auto_fetch_done = False

if not st.session_state.auto_fetch_done:
    with st.spinner("Chargement des news et analyse du sentiment..."):
        try:
            from src.ingest.auto_fetch import auto_update_all
            auto_update_all()
        except Exception as e:
            st.error(f"Erreur auto-fetch : {e}")
    st.session_state.auto_fetch_done = True
    st.rerun()

st.title("Fin Copilot TN — Banques tunisiennes")
st.caption("Analyse pedagogique du secteur bancaire tunisien. Pas un conseil en investissement.")

try:
    from src.rag.chat_tn import ask as chat_ask
    CHATBOT_OK = True
except Exception as e:
    CHATBOT_OK = False
    CHAT_ERROR = str(e)

try:
    from src.analysis.regression import prepare_dataset, run_all_regressions, analyze_by_event
    REG_OK = True
except Exception as e:
    REG_OK = False
    REG_ERROR = str(e)


@st.cache_data(ttl=60)
def load_prices():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_news():
    try:
        return pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_index():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/bank_index.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_impact():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/news_impact_clean.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_regression():
    if not REG_OK:
        return None, None
    try:
        df = prepare_dataset()
        if len(df) == 0:
            return None, None
        results = run_all_regressions(df)
        events = analyze_by_event(df)
        return df, (results, events)
    except Exception:
        return None, None


prices = load_prices()
news = load_news()
index_df = load_index()
impact = load_impact()
sent_col = "sentiment" if "sentiment" in news.columns else ("sentiment_kw" if "sentiment_kw" in news.columns else None)

st.sidebar.header("Parametres")

if len(news) > 0:
    banques_dispo = sorted(news["tickers"].unique().tolist())
else:
    banques_dispo = []

ticker = st.sidebar.selectbox("Banque", banques_dispo if banques_dispo else ["-"])

st.sidebar.markdown("---")

if st.sidebar.button("Actualiser maintenant", use_container_width=True):
    st.cache_data.clear()
    st.session_state.auto_fetch_done = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"Heure : {datetime.now().strftime('%H:%M:%S')}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Vue d'ensemble", "Detail banque", "News", "Prix",
    "Impact News", "Econometrie", "Chatbot"
])

# ===== ONGLET 1 =====
with tab1:
    st.subheader("Secteur bancaire tunisien — Vue d'ensemble")

    if len(news) == 0 or sent_col is None:
        st.warning("Aucune news.")
    else:
        stats_df = news.groupby("tickers").agg(
            sentiment_moyen=(sent_col, "mean"),
            nb_news=(sent_col, "count"),
        ).round(3).sort_values("sentiment_moyen", ascending=False)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Banques suivies", len(stats_df))
        col2.metric("News totales", len(news))
        col3.metric("Sentiment moyen", f"{news[sent_col].mean():+.3f}")
        col4.metric("Meilleure", stats_df.index[0] if len(stats_df) > 0 else "-")

        st.markdown("### Sentiment par banque")
        fig = go.Figure()
        colors = ["#27AE60" if v > 0.15 else ("#E74C3C" if v < -0.15 else "#95A5A6")
                  for v in stats_df["sentiment_moyen"]]
        fig.add_trace(go.Bar(
            x=stats_df.index, y=stats_df["sentiment_moyen"],
            marker_color=colors,
            text=[f"{v:+.3f}" for v in stats_df["sentiment_moyen"]],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(height=400, yaxis_range=[-1, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ===== ONGLET 2 =====
with tab2:
    if ticker == "-" or sent_col is None:
        st.info("Aucune banque disponible.")
    else:
        st.subheader(f"Banque : {ticker}")
        news_t = news[news["tickers"] == ticker].sort_values("date", ascending=False)

        if len(news_t) > 0:
            sent_moy = news_t[sent_col].mean()
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", f"{sent_moy:+.3f}")
            col2.metric("Nb news", len(news_t))
            col3.metric("Derniere news",
                        news_t.iloc[0]["date"].strftime("%d/%m") if len(news_t) > 0 else "-")

            price_t = prices[prices["ticker"] == ticker].sort_values("date")
            if len(price_t) > 0:
                st.metric("Dernier prix", f"{price_t.iloc[-1]['close']:.2f} DT")

            if len(price_t) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_t["date"], y=price_t["close"], mode="lines"))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### News")
            for _, r in news_t.head(30).iterrows():
                s = r[sent_col]
                icon = "[+]" if s > 0.2 else ("[-]" if s < -0.2 else "[=]")
                st.markdown(f"{icon} **{r['title'][:120]}** — `{s:+.2f}`")

# ===== ONGLET 3 =====
with tab3:
    st.subheader("Toutes les news")
    if len(news) > 0:
        df_f = news.sort_values("date", ascending=False).head(50)
        for _, r in df_f.iterrows():
            s = r[sent_col]
            icon = "[+]" if s > 0.2 else ("[-]" if s < -0.2 else "[=]")
            with st.expander(f"{icon} [{r['tickers']}] {r['title'][:100]} ({s:+.2f})"):
                st.markdown(f"**Banque** : {r['tickers']}")
                st.markdown(f"**Sentiment** : `{s:+.3f}`")
                st.markdown(f"**Type** : `{r.get('event_type', '?')}`")
                if r.get("url"):
                    st.markdown(f"[Lien]({r['url']})")

# ===== ONGLET 4 =====
with tab4:
    st.subheader("Prix des banques")
    if len(prices) > 0:
        last_prices = prices.sort_values("date").groupby("ticker").last().reset_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            for _, r in last_prices.sort_values("ticker").iterrows():
                st.metric(r["ticker"], f"{r['close']:.2f} DT")
        with col2:
            fig = go.Figure()
            for t in sorted(prices["ticker"].unique()):
                sub = prices[prices["ticker"] == t].sort_values("date")
                fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], name=t, mode="lines"))
            fig.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ===== ONGLET 5 : IMPACT NEWS =====
with tab5:
    st.subheader("📊 Impact des news sur le cours")
    st.caption("Pour chaque news : sentiment, evolution du cours, comparaison a l'indice bancaire")

    with st.expander("📚 Comment lire ce tableau ?", expanded=False):
        st.markdown("""
### 🎯 Objectif

Voir **comment le cours a réagi** après chaque news.

### 📊 Les colonnes

| Colonne | Signification |
|---|---|
| **Sentiment** | Score de la news (-1 = négatif, +1 = positif) |
| **Prix news** | Prix du cours le jour de la news |
| **Prix J+1/J+5/J+20** | Prix 1, 5, 20 jours après |
| **Var %** | Variation en % |
| **Indice** | Rendement de l'indice bancaire sur la même période |
| **Alpha** | **Surperformance** = Var news - Var indice |
| **Verdict** | Sentiment confirmé par le cours ? |

### 🎯 Alpha, c'est quoi ?

**L'alpha** mesure la **surperformance** par rapport au marché.

- **Alpha > 0** → La news a fait **mieux que le marché** ✅
- **Alpha < 0** → La news a fait **moins bien que le marché** ❌

C'est **la mesure la plus importante** en finance.
        """)

    if len(impact) == 0:
        st.warning("Aucune donnee d'impact. Lance : python -m src.analysis.news_impact_clean")
    else:
        # Stats par sentiment
        df_clean = impact[(impact["has_dividend"] == False) & (impact["var_j20_pct"].notna())]
        df_pos = df_clean[df_clean["sentiment"] > 0.1]
        df_neg = df_clean[df_clean["sentiment"] < -0.1]

        st.markdown("### 🎯 Resultats globaux (vs indice bancaire)")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 News positives")
            if len(df_pos) > 0:
                st.metric("Nombre", len(df_pos))
                st.metric("Rendement moyen J+20", f"{df_pos['var_j20_pct'].mean():+.2f}%")
                if "bench_j20_pct" in df_pos.columns:
                    st.metric("Indice bancaire J+20", f"{df_pos['bench_j20_pct'].mean():+.2f}%")
                    st.metric("Alpha moyen J+20",
                              f"{df_pos['alpha_j20_pct'].mean():+.2f} pp",
                              delta="Surperformance" if df_pos['alpha_j20_pct'].mean() > 0 else "Sous-performance")
                hit = (df_pos["var_j20_pct"] > 0).mean() * 100
                st.metric("Taux de hausse", f"{hit:.1f}%")
            else:
                st.info("Aucune news positive")

        with col2:
            st.markdown("#### 🔴 News negatives")
            if len(df_neg) > 0:
                st.metric("Nombre", len(df_neg))
                st.metric("Rendement moyen J+20", f"{df_neg['var_j20_pct'].mean():+.2f}%")
                if "bench_j20_pct" in df_neg.columns:
                    st.metric("Indice bancaire J+20", f"{df_neg['bench_j20_pct'].mean():+.2f}%")
                    st.metric("Alpha moyen J+20",
                              f"{df_neg['alpha_j20_pct'].mean():+.2f} pp",
                              delta="Sous-performance" if df_neg['alpha_j20_pct'].mean() < 0 else "Surperformance")
                hit = (df_neg["var_j20_pct"] < 0).mean() * 100
                st.metric("Taux de baisse", f"{hit:.1f}%")
            else:
                st.info("Aucune news negative")

        # Filtres
        st.markdown("---")
        st.markdown("### 📋 Toutes les news avec impact")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_ticker = st.multiselect("Banques", banques_dispo, default=banques_dispo)
        with col2:
            filter_verdict = st.multiselect(
                "Verdict",
                sorted(impact["verdict"].unique().tolist()),
                default=sorted(impact["verdict"].unique().tolist())
            )
        with col3:
            only_with_impact = st.checkbox("Seulement avec J+20 mesurable", value=True)

        df_display = impact[impact["ticker"].isin(filter_ticker)]
        df_display = df_display[df_display["verdict"].isin(filter_verdict)]
        if only_with_impact:
            df_display = df_display[df_display["var_j20_pct"].notna()]

        st.markdown(f"**{len(df_display)} news** correspondantes")

        # Tableau affichage
        display_cols = ["date", "ticker", "sentiment", "price_at_news",
                        "var_j1_pct", "var_j5_pct", "var_j20_pct",
                        "bench_j20_pct", "alpha_j20_pct", "verdict"]
        display_cols = [c for c in display_cols if c in df_display.columns]

        df_show = df_display[display_cols].copy().head(100)
        for col in ["sentiment", "price_at_news", "var_j1_pct", "var_j5_pct", "var_j20_pct",
                    "bench_j20_pct", "alpha_j20_pct"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].round(2)

        st.dataframe(df_show, use_container_width=True)

        # Exemples detailles
        st.markdown("---")
        st.markdown("### 🔍 Exemples detailles")

        for _, r in df_display.head(10).iterrows():
            s = r["sentiment"]
            icon = "🟢" if s > 0.2 else ("🔴" if s < -0.2 else "⚪")

            with st.expander(f"{icon} [{r['ticker']}] {r['date'].strftime('%d/%m/%Y')} — {r['title'][:80]}"):
                st.markdown(f"**Titre complet** : {r['title']}")
                st.markdown(f"**Sentiment** : `{s:+.3f}`")
                st.markdown(f"**Type** : `{r.get('event_type', '?')}`")
                st.markdown(f"**Verdict** : `{r['verdict']}`")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Prix news", f"{r['price_at_news']:.2f} DT")
                col2.metric("J+5", f"{r['var_j5_pct']:+.2f}%" if pd.notna(r['var_j5_pct']) else "-")
                col3.metric("J+20", f"{r['var_j20_pct']:+.2f}%" if pd.notna(r['var_j20_pct']) else "-")
                if pd.notna(r.get('alpha_j20_pct')):
                    col4.metric("Alpha J+20", f"{r['alpha_j20_pct']:+.2f} pp",
                                delta="Superforme" if r['alpha_j20_pct'] > 0 else "Sous-performe")

                if r.get("url"):
                    st.markdown(f"[Lire la news]({r['url']})")

# ===== ONGLET 6 : ECONOMETRIE =====
with tab6:
    st.subheader("📉 Analyse econometrique")

    with st.expander("📚 Glossaire : comprendre les chiffres", expanded=False):
        st.markdown("""
### 🎯 Qu'est-ce qu'on cherche ?

On teste si **le sentiment des news influence le cours** via une **régression linéaire**.

### 📊 Les 4 chiffres clés

| Indicateur | Signification | Bon si... |
|---|---|---|
| **Coefficient** | Variation du cours quand sentiment +1.0 | > 0 (positif) |
| **R²** | Qualité du modèle (0 à 1) | > 0.02 (bon en finance) |
| **P-value** | Probabilité que ce soit un hasard | < 0.05 (significatif) |
| **n** | Nombre d'observations | Plus grand = mieux |

### 🚦 Interprétation

- **p < 0.05** ✅ = effet réel
- **p > 0.05** ❌ = on ne peut pas conclure
        """)

    if not REG_OK:
        st.error(f"Module regression indisponible : {REG_ERROR}")
    else:
        with st.spinner("Calcul..."):
            reg_df, reg_output = load_regression()

        if reg_df is None:
            st.warning("Pas assez de donnees.")
        else:
            results, events = reg_output
            st.success(f"Regression sur **{len(reg_df)} observations**")

            for horizon, r in results.items():
                if "error" in r:
                    st.error(f"**{horizon}** : {r['error']}")
                    continue

                sig = r["significant"]
                sig_txt = "SIGNIFICATIF" if sig else "Non significatif"

                with st.expander(f"{horizon} — Coef : {r['slope']:+.4f} — p : {r['p_value']:.4f} — {sig_txt}", expanded=sig):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Coefficient", f"{r['slope']:+.4f}")
                    col2.metric("R2", f"{r['r_squared']:.4f}")
                    col3.metric("P-value", f"{r['p_value']:.4f}")
                    col4.metric("n", r["n"])

                    if sig:
                        st.success(f"📖 Quand le sentiment +1.0 → le cours varie de **{r['slope']*100:+.2f}%** sur **{horizon}**.")
                    else:
                        st.info(f"📖 Pas de relation significative a {horizon}.")

            st.markdown("---")
            st.markdown("### 🔬 Nuage de points")
            horizon_choice = st.selectbox("Horizon", ["J+1", "J+5", "J+20"], index=2)
            h = int(horizon_choice.replace("J+", ""))
            target = f"ret_{h}"

            sub_plot = reg_df.dropna(subset=[target, "sentiment"])
            if len(sub_plot) > 0:
                x = sub_plot["sentiment"].values
                y = sub_plot[target].values * 100

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                          marker=dict(size=6, color="#2E86DE", opacity=0.5),
                                          name="News"))
                if len(x) > 1:
                    slope, intercept, _, _, _ = stats.linregress(x, sub_plot[target].values)
                    x_line = np.linspace(x.min(), x.max(), 50)
                    y_line = (slope * x_line + intercept) * 100
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                              line=dict(color="#E74C3C", width=3),
                                              name="Regression"))
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                fig.update_layout(height=450,
                                  xaxis_title="Sentiment",
                                  yaxis_title=f"Rendement {horizon} (%)")
                st.plotly_chart(fig, use_container_width=True)

# ===== ONGLET 7 : CHATBOT =====
with tab7:
    st.subheader("Chatbot banques tunisiennes")
    if not CHATBOT_OK:
        st.error(f"Chatbot indisponible : {CHAT_ERROR}")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Pose ta question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Reflexion..."):
                    try:
                        answer = chat_ask(prompt)
                    except Exception as e:
                        answer = f"Erreur : {e}"
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button("Effacer"):
            st.session_state.messages = []
            st.rerun()

st.warning("Outil pedagogique. Pas un conseil en investissement.")