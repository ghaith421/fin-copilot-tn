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
st.caption("Analyse IA du secteur bancaire tunisien. Pas un conseil en investissement.")

# =========================================================
# IMPORTS
# =========================================================
try:
    from src.rag.chat_tn import ask as chat_ask
    CHATBOT_OK = True
except Exception as e:
    CHATBOT_OK = False
    CHAT_ERROR = str(e)

try:
    from src.analysis.news_twins import find_twins, predict_impact_from_twins
    TWINS_OK = True
except Exception as e:
    TWINS_OK = False
    TWINS_ERROR = str(e)

try:
    from src.analysis.divergence import compute_divergence
    DIVERGENCE_OK = True
except Exception as e:
    DIVERGENCE_OK = False
    DIVERGENCE_ERROR = str(e)

try:
    from src.nlp.sentiment_explain import explain_sentiment
    EXPLAIN_OK = True
except Exception as e:
    EXPLAIN_OK = False
    EXPLAIN_ERROR = str(e)


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
period_label = st.sidebar.selectbox("Periode d'analyse", list(period_options.keys()), index=2)
period_days = period_options[period_label]

st.sidebar.markdown("---")
st.sidebar.markdown("### Actualisation")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Rafraichir", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col2:
    if st.button("Tout relancer", use_container_width=True):
        st.cache_data.clear()
        st.session_state.auto_fetch_done = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Heure** : `{datetime.now().strftime('%H:%M:%S')}`")

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


# =========================================================
# FILTRE PERIODE
# =========================================================
cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)

if len(news) > 0:
    news_filtered = news[news["date"] >= cutoff].copy()
else:
    news_filtered = news

n_news_total = len(news)
n_news_filtered = len(news_filtered)


# =========================================================
# ONGLETS (8)
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Vue d'ensemble", "Detail banque", "News", "Prix",
    "Impact News", "News Twins", "Divergence", "Chatbot"
])

# ===== ONGLET 1 =====
with tab1:
    st.subheader(f"Secteur bancaire tunisien — {period_label}")

    if len(news_filtered) == 0 or sent_col is None:
        st.warning("Aucune news sur cette periode.")
    else:
        stats_df = news_filtered.groupby("tickers").agg(
            sentiment_moyen=(sent_col, "mean"),
            nb_news=(sent_col, "count"),
        ).round(3)

        for bank in ALL_BANKS:
            if bank not in stats_df.index:
                stats_df.loc[bank] = [0.0, 0]

        stats_df = stats_df.sort_values("sentiment_moyen", ascending=False)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Banques", len(stats_df))
        col2.metric("News", len(news_filtered), delta=f"{n_news_filtered}/{n_news_total}")
        col3.metric("Sentiment moyen", f"{news_filtered[sent_col].mean():+.3f}")
        active_stats = stats_df[stats_df["nb_news"] > 0]
        best = active_stats.index[0] if len(active_stats) > 0 else "-"
        col4.metric("Meilleure", best)

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
            x=stats_df.index, y=stats_df["sentiment_moyen"],
            marker_color=colors,
            text=[f"{v:+.3f}" if n > 0 else "0 news" for v, n in zip(stats_df["sentiment_moyen"], stats_df["nb_news"])],
            textposition="outside",
        ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(height=400, yaxis_range=[-1, 1], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tableau recapitulatif")
        df_show = stats_df.copy()
        df_show.columns = ["Sentiment", "Nb news"]
        df_show["Statut"] = df_show.apply(
            lambda r: "Aucune news" if r["Nb news"] == 0 else (
                "Positif" if r["Sentiment"] > 0.15 else (
                    "Negatif" if r["Sentiment"] < -0.15 else "Neutre"
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


# ===== ONGLET 3 : NEWS + EXPLICATION =====
with tab3:
    st.subheader(f"Toutes les news ({period_label})")
    st.caption(f"Cliquez sur une news pour voir l'**explication du sentiment** (XAI)")

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

                # Explication XAI
                if EXPLAIN_OK:
                    with st.spinner("Analyse du sentiment..."):
                        text_full = (r.get("title", "") or "") + ". " + (r.get("summary", "") or "")
                        exp = explain_sentiment(text_full)
                        st.markdown(f"**🔍 Explication (XAI)** : {exp['summary']}")
                        if exp["contributions"]:
                            st.markdown("**Mots clés détectés :**")
                            for c in exp["contributions"][:5]:
                                emoji = "🟢" if c["weight"] > 0 else "🔴"
                                intens = f" *(intensifié par '{c['intensifier']}')*" if c.get("intensifier") else ""
                                st.markdown(f"- {emoji} `{c['word']}` → **{c['weight']:+.2f}**{intens}")

                if r.get("url"):
                    st.markdown(f"[Lire la news]({r['url']})")
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
            fig.update_layout(height=500, hovermode="x unified", title=f"Evolution sur {period_label}")
            st.plotly_chart(fig, use_container_width=True)


# ===== ONGLET 5 : IMPACT NEWS =====
with tab5:
    st.subheader("Impact des news")
    st.caption("Analyse sur TOUTE la base")

    impact_full = impact.copy() if len(impact) > 0 else pd.DataFrame()

    if len(impact_full) == 0:
        st.warning("Aucune donnee d'impact.")
    else:
        df_clean = impact_full[impact_full["has_dividend"] == False]

        horizon = st.radio("Horizon", ["J+1", "J+5", "J+20"], index=1, horizontal=True)
        h = int(horizon.replace("J+", ""))
        var_col = f"var_j{h}_pct"
        bench_col = f"bench_j{h}_pct"
        alpha_col = f"alpha_j{h}_pct"

        df_h = df_clean[df_clean[var_col].notna()]
        df_pos = df_h[df_h["sentiment"] > 0.1]
        df_neg = df_h[df_h["sentiment"] < -0.1]

        st.info(f"**{len(df_h)}** news avec impact **{horizon}** sur **{len(impact_full)}** au total")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### 🟢 News positives (n={len(df_pos)})")
            if len(df_pos) > 0:
                st.metric(f"Rendement {horizon}", f"{df_pos[var_col].mean():+.2f}%")
                if alpha_col in df_pos.columns:
                    alpha_val = df_pos[alpha_col].mean()
                    delta_val = "Surperformance" if alpha_val > 0 else "Sous-performance"
                    st.metric(f"Alpha {horizon}", f"{alpha_val:+.2f} pp", delta=delta_val)

        with col2:
            st.markdown(f"#### 🔴 News negatives (n={len(df_neg)})")
            if len(df_neg) > 0:
                st.metric(f"Rendement {horizon}", f"{df_neg[var_col].mean():+.2f}%")
                if alpha_col in df_neg.columns:
                    alpha_val = df_neg[alpha_col].mean()
                    delta_val = "Surperformance" if alpha_val > 0 else "Sous-performance"
                    st.metric(f"Alpha {horizon}", f"{alpha_val:+.2f} pp", delta=delta_val)

        st.markdown("---")
        st.markdown(f"### Toutes les news avec impact {horizon}")
        df_display = df_h.sort_values("date", ascending=False).head(100)
        display_cols = ["date", "ticker", "sentiment", "price_at_news", var_col, bench_col, alpha_col, "verdict"]
        display_cols = [c for c in display_cols if c in df_display.columns]
        if len(df_display) > 0:
            st.dataframe(df_display[display_cols], use_container_width=True)


# ===== ONGLET 6 : NEWS TWINS =====
with tab6:
    st.subheader("🔍 News Twin Finder")
    st.caption("Trouve les news passées similaires et prédit l'impact futur")

    if not TWINS_OK:
        st.error(f"Module indisponible : {TWINS_ERROR}")
        st.info("Lance d'abord : `python -m src.analysis.news_twins --build`")
    else:
        with st.expander("Comment ça marche ?", expanded=False):
            st.markdown("""
**Principe** : quand une nouvelle news arrive, on cherche les **news passées similaires** (via embeddings) et on regarde **ce qui s'est passé après**.

**Utilité** : prédire l'impact d'une news **avant** que le marché ne réagisse.

**Exemple** :
- News : "BIAT annonce un PNB en hausse"
- Jumeaux : 5 news similaires passées
- Impact moyen de ces jumeaux : +2.3% à J+5
- Prédiction : cette news pourrait faire +2.3% à J+5
            """)

        # Choix
        mode = st.radio("Mode", ["Choisir une news existante", "Saisir une news"], horizontal=True)

        if mode == "Choisir une news existante":
            if len(news) > 0:
                # Prendre les 50 dernières news
                recent_news = news.sort_values("date", ascending=False).head(50)
                options = [f"[{r['tickers']}] {r['title'][:80]}" for _, r in recent_news.iterrows()]
                selected_idx = st.selectbox("News", range(len(options)), format_func=lambda i: options[i])

                if selected_idx is not None:
                    row = recent_news.iloc[selected_idx]
                    query_text = (row.get("title", "") or "") + ". " + (row.get("summary", "") or "")
                    ticker_query = row["tickers"]

                    if st.button("🔍 Trouver les jumeaux", type="primary"):
                        with st.spinner("Recherche des jumeaux..."):
                            result = predict_impact_from_twins(query_text, ticker=ticker_query, k=5)

                        if result is None:
                            st.warning("Aucun jumeau trouve")
                        else:
                            st.success(f"**{result['n_twins']}** jumeaux trouves")

                            col1, col2, col3 = st.columns(3)
                            if result["j1_mean"] is not None:
                                col1.metric("Impact J+1 moyen", f"{result['j1_mean']:+.2f}%",
                                            delta=f"Hit rate {result['j1_hit_rate']:.0f}%")
                            if result["j5_mean"] is not None:
                                col2.metric("Impact J+5 moyen", f"{result['j5_mean']:+.2f}%",
                                            delta=f"Hit rate {result['j5_hit_rate']:.0f}%")
                            if result["j20_mean"] is not None:
                                col3.metric("Impact J+20 moyen", f"{result['j20_mean']:+.2f}%")

                            st.markdown("### 📋 Jumeaux trouvés")
                            for _, t in result["twins"].iterrows():
                                sim = t.get("similarity", 0)
                                j5 = t.get("var_j5_pct", np.nan)
                                j5_str = f"{j5:+.2f}%" if pd.notna(j5) else "N/A"
                                st.markdown(f"**[{sim:.3f}]** {t['title'][:100]}")
                                st.caption(f"Impact J+5 : `{j5_str}` | Sentiment : `{t.get('sentiment', 0):+.2f}`")

        else:
            # Saisie manuelle
            text_input = st.text_area("Titre de la news (ou description)",
                                       placeholder="Ex: BIAT annonce un PNB en hausse de 15%",
                                       height=100)
            ticker_input = st.selectbox("Banque (optionnel)", ["Toutes"] + sorted(news["tickers"].unique().tolist()))

            if st.button("🔍 Prédire l'impact", type="primary") and text_input:
                with st.spinner("Analyse..."):
                    tk = None if ticker_input == "Toutes" else ticker_input
                    result = predict_impact_from_twins(text_input, ticker=tk, k=5)

                if result is None:
                    st.warning("Aucun jumeau trouve")
                else:
                    st.success(f"**{result['n_twins']}** jumeaux trouves")

                    col1, col2, col3 = st.columns(3)
                    if result["j1_mean"] is not None:
                        col1.metric("Impact J+1 moyen", f"{result['j1_mean']:+.2f}%")
                    if result["j5_mean"] is not None:
                        col2.metric("Impact J+5 moyen", f"{result['j5_mean']:+.2f}%")
                    if result["j20_mean"] is not None:
                        col3.metric("Impact J+20 moyen", f"{result['j20_mean']:+.2f}%")

                    st.markdown("### 📋 Jumeaux trouvés")
                    for _, t in result["twins"].iterrows():
                        sim = t.get("similarity", 0)
                        j5 = t.get("var_j5_pct", np.nan)
                        j5_str = f"{j5:+.2f}%" if pd.notna(j5) else "N/A"
                        st.markdown(f"**[{sim:.3f}]** {t['title'][:100]}")
                        st.caption(f"Impact J+5 : `{j5_str}` | Sentiment : `{t.get('sentiment', 0):+.2f}`")


# ===== ONGLET 7 : DIVERGENCE =====
with tab7:
    st.subheader("🚨 Signal Divergence")
    st.caption("Détecte les écarts entre sentiment et cours (opportunités)")

    if not DIVERGENCE_OK:
        st.error(f"Module indisponible : {DIVERGENCE_ERROR}")
    else:
        with st.expander("Comment ça marche ?", expanded=False):
            st.markdown("""
**Principe** : quand le sentiment et le cours **divergent**, il y a potentiellement une opportunité.

- 🟢 **Achat potentiel** : sentiment positif MAIS cours pas encore monté
- 🔴 **Vente potentielle** : sentiment négatif MAIS cours pas encore baissé
- ⚪ **Cohérent** : sentiment et cours alignés

**Historique** : les marchés mettent 1-3 semaines à intégrer pleinement l'information.
            """)

        window = st.radio("Fenetre d'analyse", ["7 jours", "14 jours", "30 jours"], index=2, horizontal=True)
        window_days = int(window.split()[0])

        min_news = st.slider("Minimum de news par banque", 1, 5, 2)

        with st.spinner("Analyse des divergences..."):
            df_div = compute_divergence(window_days=window_days, min_news=min_news)

        if len(df_div) == 0:
            st.warning(f"Aucune divergence detectee (pas assez de donnees sur {window})")
        else:
            # Alertes en haut
            alerts = df_div[df_div["type"] != "Coherent"]
            if len(alerts) > 0:
                for _, a in alerts.iterrows():
                    if a["type"] == "Achat potentiel":
                        st.success(f"🟢 **{a['ticker']}** — Sentiment **{a['sentiment']:+.2f}** mais cours **{a['var_cours_pct']:+.2f}%** — Divergence **{a['divergence']:+.2f}**")
                    elif a["type"] == "Vente potentielle":
                        st.error(f"🔴 **{a['ticker']}** — Sentiment **{a['sentiment']:+.2f}** mais cours **{a['var_cours_pct']:+.2f}%** — Divergence **{a['divergence']:+.2f}**")

            st.markdown("---")
            st.markdown(f"### 📊 Analyse complete sur {window}")
            st.dataframe(df_div[["ticker", "sentiment", "var_cours_pct", "nb_news", "divergence", "type"]],
                         use_container_width=True)

            # Graphique
            fig = go.Figure()
            colors = ["#27AE60" if t == "Achat potentiel" else ("#E74C3C" if t == "Vente potentielle" else "#95A5A6")
                      for t in df_div["type"]]
            fig.add_trace(go.Bar(
                x=df_div["ticker"],
                y=df_div["divergence"],
                marker_color=colors,
                text=[f"{v:+.2f}" for v in df_div["divergence"]],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.add_hline(y=0.4, line_dash="dash", line_color="#27AE60", line_width=1)
            fig.add_hline(y=-0.4, line_dash="dash", line_color="#E74C3C", line_width=1)
            fig.update_layout(height=400, title=f"Divergence par banque ({window})",
                              yaxis_title="Divergence", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ===== ONGLET 8 : CHATBOT =====
with tab8:
    st.subheader("Chatbot banques tunisiennes")
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