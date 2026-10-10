import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from config import DATA_RAW

# =========================================================
# CONFIGURATION PAGE
# =========================================================
st.set_page_config(page_title="Fin Copilot TN", layout="wide", page_icon="🇹🇳")

# =========================================================
# AUTO-FETCH (UNE SEULE FOIS)
# =========================================================
if "auto_fetch_done" not in st.session_state:
    st.session_state.auto_fetch_done = False

if not st.session_state.auto_fetch_done:
    with st.spinner("⏳ Chargement des news et analyse du sentiment (peut prendre 10-15 min)..."):
        try:
            from src.ingest.auto_fetch import auto_update_all
            auto_update_all()
        except Exception as e:
            st.error(f"Erreur auto-fetch : {e}")
    st.session_state.auto_fetch_done = True
    st.rerun()

st.title("🇹🇳 Fin Copilot TN — Banques tunisiennes")
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


# =========================================================
# CHARGEMENT DES DONNEES
# =========================================================
@st.cache_data(ttl=60)
def load_prices():
    try:
        df = pd.read_parquet(f"{DATA_RAW}/bvmt_prices.parquet")
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_news():
    try:
        return pd.read_parquet(f"{DATA_RAW}/banks_news_scored.parquet")
    except Exception:
        try:
            return pd.read_parquet(f"{DATA_RAW}/banks_news.parquet")
        except Exception:
            return pd.DataFrame()


prices = load_prices()
news = load_news()
sent_col = "sentiment" if "sentiment" in news.columns else ("sentiment_kw" if "sentiment_kw" in news.columns else None)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Parametres")

if len(news) > 0:
    banques_dispo = sorted(news["tickers"].unique().tolist())
else:
    banques_dispo = []

ticker = st.sidebar.selectbox("🏦 Banque", banques_dispo if banques_dispo else ["—"])

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Actualiser maintenant", use_container_width=True):
    st.cache_data.clear()
    st.session_state.auto_fetch_done = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Heure** : {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.info("⚪ Auto-refresh desactive")


# =========================================================
# ONGLETS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble", "🏦 Detail banque", "📰 News", "📈 Prix", "💬 Chatbot"
])


# =========================================================
# ONGLET 1 : VUE D'ENSEMBLE
# =========================================================
with tab1:
    st.subheader("📊 Secteur bancaire tunisien — Vue d'ensemble")

    if len(news) == 0 or sent_col is None:
        st.warning("Aucune news. Attends le rafraichissement automatique...")
    else:
        stats = news.groupby("tickers").agg(
            sentiment_moyen=(sent_col, "mean"),
            nb_news=(sent_col, "count"),
            positifs=(sent_col, lambda s: (s > 0.2).sum()),
            negatifs=(sent_col, lambda s: (s < -0.2).sum()),
        ).round(3).sort_values("sentiment_moyen", ascending=False)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏦 Banques suivies", len(stats))
        col2.metric("📰 News totales", len(news))
        col3.metric("🟢 Sentiment moyen", f"{news[sent_col].mean():+.3f}")
        meilleure = stats.index[0] if len(stats) > 0 else "—"
        col4.metric("🥇 Meilleure", meilleure)

        st.markdown("### 🎯 Sentiment par banque")
        fig = go.Figure()
        colors = ["#27AE60" if v > 0.15 else ("#E74C3C" if v < -0.15 else "#95A5A6")
                  for v in stats["sentiment_moyen"]]

        fig.add_trace(go.Bar(
            x=stats.index,
            y=stats["sentiment_moyen"],
            marker_color=colors,
            text=[f"{v:+.3f}" for v in stats["sentiment_moyen"]],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(height=400, yaxis_range=[-1, 1],
                          margin=dict(l=10, r=10, t=30, b=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Tableau recapitulatif")
        df_show = stats.copy()
        df_show.columns = ["Sentiment moyen", "Nb news", "Positifs 🟢", "Negatifs 🔴"]
        df_show["Tendance"] = df_show["Sentiment moyen"].apply(
            lambda x: "🟢 Positif" if x > 0.15 else ("🔴 Negatif" if x < -0.15 else "⚪ Neutre")
        )
        st.dataframe(df_show, use_container_width=True)


# =========================================================
# ONGLET 2 : DETAIL BANQUE
# =========================================================
with tab2:
    if ticker == "—" or sent_col is None:
        st.info("Aucune banque disponible.")
    else:
        st.subheader(f"🏦 {ticker}")
        news_t = news[news["tickers"] == ticker].sort_values("date", ascending=False)

        if len(news_t) > 0:
            sent_moy = news_t[sent_col].mean()
            emoji = "🟢" if sent_moy > 0.15 else ("🔴" if sent_moy < -0.15 else "⚪")

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{emoji} Sentiment", f"{sent_moy:+.3f}")
            col2.metric("📰 Nb news", len(news_t))
            col3.metric("📅 Derniere news",
                        news_t.iloc[0]["date"].strftime("%d/%m") if len(news_t) > 0 else "—")

            price_t = prices[prices["ticker"] == ticker].sort_values("date")
            if len(price_t) > 0:
                last_price = price_t.iloc[-1]["close"]
                st.metric("💰 Dernier prix", f"{last_price:.2f} DT")

            if len(price_t) > 1:
                st.markdown("### 📈 Evolution du prix")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_t["date"], y=price_t["close"],
                    mode="lines+markers", name="Prix",
                    line=dict(color="#2E86DE", width=2),
                ))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📰 Toutes les news")
            for _, r in news_t.iterrows():
                s = r[sent_col]
                icon = "🟢" if s > 0.2 else ("🔴" if s < -0.2 else "⚪")
                title = r["title"][:120]
                date_str = r["date"].strftime("%d/%m") if pd.notna(r["date"]) else "?"
                event = r.get("event_type", "")
                st.markdown(
                    f"{icon} **{title}**  \n"
                    f"<small>`{s:+.2f}` · *{event}* · {date_str}</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aucune news pour cette banque.")


# =========================================================
# ONGLET 3 : NEWS
# =========================================================
with tab3:
    st.subheader("📰 Toutes les news bancaires")

    if len(news) == 0 or sent_col is None:
        st.warning("Aucune news.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_bank = st.multiselect("Banques", banques_dispo, default=banques_dispo)
        with col2:
            filter_sent = st.selectbox("Sentiment", ["Tous", "Positifs", "Negatifs", "Neutres"])
        with col3:
            events_list = sorted(news["event_type"].dropna().unique().tolist()) if "event_type" in news.columns else ["Tous"]
            filter_event = st.selectbox("Type d'evenement", ["Tous"] + events_list)

        df_f = news[news["tickers"].isin(filter_bank)]
        if filter_sent == "Positifs":
            df_f = df_f[df_f[sent_col] > 0.2]
        elif filter_sent == "Negatifs":
            df_f = df_f[df_f[sent_col] < -0.2]
        elif filter_sent == "Neutres":
            df_f = df_f[(df_f[sent_col] >= -0.2) & (df_f[sent_col] <= 0.2)]

        if filter_event != "Tous" and "event_type" in df_f.columns:
            df_f = df_f[df_f["event_type"] == filter_event]

        st.markdown(f"**{len(df_f)} news** correspondantes")
        df_f = df_f.sort_values("date", ascending=False).head(50)

        for _, r in df_f.iterrows():
            s = r[sent_col]
            icon = "🟢" if s > 0.2 else ("🔴" if s < -0.2 else "⚪")
            with st.expander(f"{icon} [{r['tickers']}] {r['title'][:100]}  ({s:+.2f})"):
                st.markdown(f"**Titre** : {r['title']}")
                st.markdown(f"**Banque** : {r['tickers']}")
                st.markdown(f"**Date** : {r['date']}")
                st.markdown(f"**Sentiment** : `{s:+.3f}`")
                st.markdown(f"**Type** : `{r.get('event_type', '?')}`")
                if r.get("raison"):
                    st.markdown(f"**Raison** : {r['raison']}")
                if r.get("url"):
                    st.markdown(f"[Lien]({r['url']})")


# =========================================================
# ONGLET 4 : PRIX
# =========================================================
with tab4:
    st.subheader("📈 Prix des banques tunisiennes")

    if len(prices) == 0:
        st.warning("Aucun prix. Attends le rafraichissement automatique...")
    else:
        last_prices = prices.sort_values("date").groupby("ticker").last().reset_index()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 💰 Derniers cours")
            for _, r in last_prices.sort_values("ticker").iterrows():
                st.metric(r["ticker"], f"{r['close']:.2f} DT")

        with col2:
            st.markdown("### 📊 Comparaison")
            fig = go.Figure()
            for t in sorted(prices["ticker"].unique()):
                sub = prices[prices["ticker"] == t].sort_values("date")
                if len(sub) > 0:
                    fig.add_trace(go.Scatter(
                        x=sub["date"], y=sub["close"], name=t, mode="lines+markers",
                    ))
            fig.update_layout(height=500, hovermode="x unified",
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Historique")
        st.dataframe(prices.sort_values("date", ascending=False), use_container_width=True)


# =========================================================
# ONGLET 5 : CHATBOT
# =========================================================
with tab5:
    st.subheader("💬 Assistant banques tunisiennes")

    if not CHATBOT_OK:
        st.error(f"Chatbot indisponible : {CHAT_ERROR}")
    else:
        st.markdown("**Exemples :**")
        cols = st.columns(4)
        quick_qs = [
            "Quelle banque a le meilleur sentiment ?",
            "Analyse BIAT",
            "Compare BIAT et UBCI",
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

        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()


st.warning("⚠️ Outil pedagogique. Ne constitue pas un conseil en investissement.")