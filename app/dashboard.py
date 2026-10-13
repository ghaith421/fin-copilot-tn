import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta
from config import DATA_RAW

st.set_page_config(page_title="Fin Copilot TN", layout="wide", page_icon="🇹🇳")

# =========================================================
# AUTO-FETCH
# =========================================================
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

# =========================================================
# IMPORT CHATBOT
# =========================================================
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


# =========================================================
# CHARGEMENT
# =========================================================
@st.cache_data(ttl=60)
def load_prices():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/bvmt_history.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        try:
            df = pd.read_parquet(f"{DATA_RAW}/bvmt_prices.parquet")
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=60)
def load_news():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        try:
            df = pd.read_parquet(f"{DATA_RAW}/banks_news.parquet")
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            return df
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=60)
def load_impact():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/news_impact_clean.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


prices = load_prices()
news = load_news()
impact = load_impact()
sent_col = "sentiment" if "sentiment" in news.columns else ("sentiment_kw" if "sentiment_kw" in news.columns else None)

# Liste complete des banques
ALL_BANKS = ["BIAT", "ATB", "AB", "AMEN", "UIB", "BNA", "BH", "STB", "BT", "BTE", "ZB", "UBCI", "SECTEUR"]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Parametres")

if len(news) > 0:
    banques_dispo = sorted(news["tickers"].unique().tolist())
else:
    banques_dispo = []

ticker = st.sidebar.selectbox("Banque", banques_dispo if banques_dispo else ["-"])

st.sidebar.markdown("---")
period_options = {
    "Aujourd'hui": 1,
    "3 jours": 3,
    "7 jours (recommande)": 7,
    "14 jours": 14,
    "30 jours": 30,
    "60 jours": 60,
    "Tout": 99999,
}
period_label = st.sidebar.selectbox(
    "Periode d'analyse",
    list(period_options.keys()),
    index=2,
)
period_days = period_options[period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### Actualisation")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Rafraichir", use_container_width=True, help="Recharge l'affichage"):
        st.cache_data.clear()
        st.rerun()

with col2:
    if st.button("Tout relancer", use_container_width=True, help="Scrape a nouveau (1-2 min)"):
        st.cache_data.clear()
        st.session_state.auto_fetch_done = False
        st.rerun()

st.sidebar.markdown("---")

now = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown(f"**Heure** : `{now}`")

news_path = f"{DATA_RAW}/banks_news_scored.parquet"
if os.path.exists(news_path):
    mtime = datetime.fromtimestamp(os.path.getmtime(news_path))
    age_min = (datetime.now() - mtime).total_seconds() / 60
    if age_min < 5:
        st.sidebar.success(f"Donnees fraiches ({age_min:.0f} min)")
    elif age_min < 60:
        st.sidebar.info(f"Donnees : {age_min:.0f} min")
    else:
        st.sidebar.warning(f"Donnees : {age_min/60:.1f} h")
else:
    st.sidebar.error("Pas de donnees")


# =========================================================
# FILTRE PAR PERIODE
# =========================================================
cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)

if len(news) > 0:
    news_filtered = news[news["date"] >= cutoff].copy()
else:
    news_filtered = news

if len(impact) > 0:
    impact_filtered = impact[impact["date"] >= cutoff].copy()
else:
    impact_filtered = impact

n_news_total = len(news)
n_news_filtered = len(news_filtered)


# =========================================================
# ONGLETS (6)
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vue d'ensemble", "Detail banque", "News", "Prix",
    "Impact News", "Chatbot"
])

# ===== ONGLET 1 =====
with tab1:
    st.subheader(f"Secteur bancaire tunisien — {period_label}")

    if len(news_filtered) == 0 or sent_col is None:
        st.warning(f"Aucune news sur cette periode. Essaie une periode plus large.")
    else:
        stats_df = news_filtered.groupby("tickers").agg(
            sentiment_moyen=(sent_col, "mean"),
            nb_news=(sent_col, "count"),
        ).round(3)

        # Ajouter TOUTES les banques (meme sans news)
        for bank in ALL_BANKS:
            if bank not in stats_df.index:
                stats_df.loc[bank] = [0.0, 0]

        stats_df = stats_df.sort_values("sentiment_moyen", ascending=False)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Banques", len(stats_df))
        col2.metric("News", len(news_filtered),
                    delta=f"{n_news_filtered}/{n_news_total}")
        col3.metric("Sentiment moyen", f"{news_filtered[sent_col].mean():+.3f}")
        active_stats = stats_df[stats_df["nb_news"] > 0]
        best = active_stats.index[0] if len(active_stats) > 0 else "-"
        col4.metric("Meilleure", best)

        st.markdown(f"### Sentiment par banque ({period_label})")

        fig = go.Figure()
        colors = []
        for _, row in stats_df.iterrows():
            if row["nb_news"] == 0:
                colors.append("#CCCCCC")
            elif row["sentiment_moyen"] > 0.15:
                colors.append("#27AE60")
            elif row["sentiment_moyen"] < -0.15:
                colors.append("#E74C3C")
            else:
                colors.append("#95A5A6")

        fig.add_trace(go.Bar(
            x=stats_df.index,
            y=stats_df["sentiment_moyen"],
            marker_color=colors,
            text=[f"{v:+.3f}" if n > 0 else "0 news" for v, n in zip(stats_df["sentiment_moyen"], stats_df["nb_news"])],
            textposition="outside",
            customdata=stats_df["nb_news"],
            hovertemplate="<b>%{x}</b><br>Sentiment : %{y:+.3f}<br>News : %{customdata}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(height=400, yaxis_range=[-1, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tableau recapitulatif")
        df_show = stats_df.copy()
        df_show.columns = ["Sentiment moyen", "Nb news"]
        df_show["Statut"] = df_show.apply(
            lambda r: "Aucune news" if r["Nb news"] == 0 else (
                "Positif" if r["Sentiment moyen"] > 0.15 else (
                    "Negatif" if r["Sentiment moyen"] < -0.15 else "Neutre"
                )
            ), axis=1
        )
        st.dataframe(df_show, use_container_width=True)

# ===== ONGLET 2 =====
with tab2:
    if ticker == "-" or sent_col is None:
        st.info("Aucune banque disponible.")
    else:
        st.subheader(f"Banque : {ticker} — {period_label}")
        news_t = news_filtered[news_filtered["tickers"] == ticker].sort_values("date", ascending=False)

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

            st.markdown(f"### News ({period_label})")
            for _, r in news_t.head(30).iterrows():
                s = r[sent_col]
                icon = "[+]" if s > 0.2 else ("[-]" if s < -0.2 else "[=]")
                st.markdown(f"{icon} **{r['title'][:120]}** — `{s:+.2f}`")
        else:
            st.info(f"Aucune news pour {ticker} sur cette periode.")

# ===== ONGLET 3 =====
with tab3:
    st.subheader(f"Toutes les news ({period_label})")
    st.caption(f"Affichage : {len(news_filtered)} news sur {period_label} (base totale : {n_news_total})")

    if len(news_filtered) > 0:
        df_f = news_filtered.sort_values("date", ascending=False).head(100)
        for _, r in df_f.iterrows():
            s = r[sent_col]
            icon = "[+]" if s > 0.2 else ("[-]" if s < -0.2 else "[=]")
            date_str = r["date"].strftime("%d/%m") if pd.notna(r["date"]) else "?"
            with st.expander(f"{icon} [{r['tickers']}] {r['title'][:100]} ({s:+.2f}) — {date_str}"):
                st.markdown(f"**Banque** : {r['tickers']}")
                st.markdown(f"**Date** : {r['date']}")
                st.markdown(f"**Sentiment** : `{s:+.3f}`")
                st.markdown(f"**Type** : `{r.get('event_type', '?')}`")
                if r.get("url"):
                    st.markdown(f"[Lien]({r['url']})")
    else:
        st.warning(f"Aucune news sur {period_label}.")

# ===== ONGLET 4 =====
with tab4:
    st.subheader("Prix des banques")
    if len(prices) > 0:
        prices_filtered = prices[prices["date"] >= cutoff]
        if len(prices_filtered) == 0:
            prices_filtered = prices

        last_prices = prices_filtered.sort_values("date").groupby("ticker").last().reset_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            for _, r in last_prices.sort_values("ticker").iterrows():
                st.metric(r["ticker"], f"{r['close']:.2f} DT")
        with col2:
            fig = go.Figure()
            for t in sorted(prices_filtered["ticker"].unique()):
                sub = prices_filtered[prices_filtered["ticker"] == t].sort_values("date")
                fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], name=t, mode="lines"))
            fig.update_layout(height=500, hovermode="x unified",
                              title=f"Evolution sur {period_label}")
            st.plotly_chart(fig, use_container_width=True)

# ===== ONGLET 5 : IMPACT NEWS =====
with tab5:
    st.subheader("Impact des news")
    st.caption("Analyse sur TOUTE la base (l'impact necessite du recul)")

    with st.expander("Pourquoi la periode est differente ici ?", expanded=False):
        st.markdown("""
**Important** : l'impact d'une news se mesure **plusieurs jours apres** sa publication.

- **J+1** : 1 jour apres (disponible rapidement)
- **J+5** : 5 jours apres
- **J+20** : 20 jours apres (necessite beaucoup de recul)

**Plus la base grandit, plus on aura d'impact J+20.**

Cet onglet affiche toujours la **base complete**.
        """)

    impact_full = impact.copy() if len(impact) > 0 else pd.DataFrame()

    if len(impact_full) == 0:
        st.warning("Aucune donnee d'impact. Lance : python -m src.analysis.news_impact_clean")
    else:
        df_clean = impact_full[impact_full["has_dividend"] == False]

        st.markdown("### Horizon d'analyse")
        horizon = st.radio(
            "Choisir l'horizon",
            ["J+1 (1 jour)", "J+5 (5 jours)", "J+20 (20 jours)"],
            index=1,
            horizontal=True,
        )

        h = 1 if "J+1" in horizon else (5 if "J+5" in horizon else 20)
        var_col = f"var_j{h}_pct"
        bench_col = f"bench_j{h}_pct"
        alpha_col = f"alpha_j{h}_pct"

        df_h = df_clean[df_clean[var_col].notna()]
        df_pos = df_h[df_h["sentiment"] > 0.1]
        df_neg = df_h[df_h["sentiment"] < -0.1]

        n_total = len(impact_full)
        n_with = len(df_h)
        st.info(f"**{n_with}** news avec impact **{horizon}** mesurable sur **{n_total}** au total")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🟢 News positives (n={len(df_pos)})")
            if len(df_pos) > 0:
                st.metric(f"Rendement {horizon}", f"{df_pos[var_col].mean():+.2f}%")
                if bench_col in df_pos.columns:
                    st.metric("Indice bancaire", f"{df_pos[bench_col].mean():+.2f}%")
                if alpha_col in df_pos.columns:
                    alpha_val = df_pos[alpha_col].mean()
                    delta_val = "Surperformance" if alpha_val > 0 else "Sous-performance"
                    st.metric(f"Alpha {horizon}", f"{alpha_val:+.2f} pp", delta=delta_val)
            else:
                st.info("Aucune news positive sur cet horizon")

        with col2:
            st.markdown(f"#### 🔴 News negatives (n={len(df_neg)})")
            if len(df_neg) > 0:
                st.metric(f"Rendement {horizon}", f"{df_neg[var_col].mean():+.2f}%")
                if bench_col in df_neg.columns:
                    st.metric("Indice bancaire", f"{df_neg[bench_col].mean():+.2f}%")
                if alpha_col in df_neg.columns:
                    alpha_val = df_neg[alpha_col].mean()
                    delta_val = "Surperformance" if alpha_val > 0 else "Sous-performance"
                    st.metric(f"Alpha {horizon}", f"{alpha_val:+.2f} pp", delta=delta_val)
            else:
                st.info("Aucune news negative sur cet horizon")

        st.markdown("---")
        st.markdown(f"### Toutes les news avec impact {horizon}")

        df_display = df_h.sort_values("date", ascending=False).head(100)
        display_cols = ["date", "ticker", "sentiment", "price_at_news",
                        var_col, bench_col, alpha_col, "verdict"]
        display_cols = [c for c in display_cols if c in df_display.columns]

        if len(df_display) > 0:
            st.dataframe(df_display[display_cols], use_container_width=True)
        else:
            st.warning(f"Aucune news avec impact {horizon} mesurable.")

# ===== ONGLET 6 : CHATBOT =====
with tab6:
    st.subheader("Chatbot banques tunisiennes")
    st.caption("Pose des questions sur les banques, la finance, ou les analyses")

    if not CHATBOT_OK:
        st.error(f"Chatbot indisponible : {CHAT_ERROR}")
    else:
        st.markdown("**Exemples :**")
        cols = st.columns(4)
        quick_qs = [
            "Quelle banque a le meilleur sentiment ?",
            "Analyse BIAT",
            "Explique la regression",
            "C'est quoi le PNB ?",
        ]
        for i, q in enumerate(quick_qs):
            with cols[i]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.setdefault("messages", [])
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.spinner("Reflexion..."):
                        try:
                            rep = chat_ask(q)
                        except Exception as e:
                            rep = f"Erreur : {e}"
                    st.session_state.messages.append({"role": "assistant", "content": rep})
                    st.rerun()

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

        if st.button("Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

st.warning("Outil pedagogique. Pas un conseil en investissement.")