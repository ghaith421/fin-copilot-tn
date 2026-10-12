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
    except Exception as e:
        print(f"[reg] Erreur : {e}")
        return None, None


prices = load_prices()
news = load_news()
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vue d'ensemble", "Detail banque", "News", "Prix",
    "Econometrie", "Chatbot"
])

# ===== ONGLET 1 =====
with tab1:
    st.subheader("Secteur bancaire tunisien — Vue d'ensemble")

    if len(news) == 0 or sent_col is None:
        st.warning("Aucune news. Attends le rafraichissement.")
    else:
        stats_df = news.groupby("tickers").agg(
            sentiment_moyen=(sent_col, "mean"),
            nb_news=(sent_col, "count"),
            positifs=(sent_col, lambda s: (s > 0.2).sum()),
            negatifs=(sent_col, lambda s: (s < -0.2).sum()),
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

        st.markdown("### Tableau recapitulatif")
        st.dataframe(stats_df, use_container_width=True)

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
                st.markdown("### Evolution du prix")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=price_t["date"], y=price_t["close"],
                                         mode="lines", name="Prix"))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### News")
            for _, r in news_t.head(30).iterrows():
                s = r[sent_col]
                icon = "[+]" if s > 0.2 else ("[-]" if s < -0.2 else "[=]")
                st.markdown(f"{icon} **{r['title'][:120]}** — `{s:+.2f}` ({r.get('event_type', '')})")

# ===== ONGLET 3 =====
with tab3:
    st.subheader("Toutes les news bancaires")
    if len(news) == 0:
        st.warning("Aucune news.")
    else:
        df_f = news.sort_values("date", ascending=False).head(50)
        for _, r in df_f.iterrows():
            s = r[sent_col]
            icon = "[+]" if s > 0.2 else ("[-]" if s < -0.2 else "[=]")
            with st.expander(f"{icon} [{r['tickers']}] {r['title'][:100]} ({s:+.2f})"):
                st.markdown(f"**Titre** : {r['title']}")
                st.markdown(f"**Banque** : {r['tickers']}")
                st.markdown(f"**Sentiment** : `{s:+.3f}`")
                st.markdown(f"**Type** : `{r.get('event_type', '?')}`")
                if r.get("url"):
                    st.markdown(f"[Lien]({r['url']})")

# ===== ONGLET 4 =====
with tab4:
    st.subheader("Prix des banques")
    if len(prices) == 0:
        st.warning("Aucun prix.")
    else:
        last_prices = prices.sort_values("date").groupby("ticker").last().reset_index()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Derniers cours")
            for _, r in last_prices.sort_values("ticker").iterrows():
                st.metric(r["ticker"], f"{r['close']:.2f} DT")
        with col2:
            fig = go.Figure()
            for t in sorted(prices["ticker"].unique()):
                sub = prices[prices["ticker"] == t].sort_values("date")
                fig.add_trace(go.Scatter(x=sub["date"], y=sub["close"], name=t, mode="lines"))
            fig.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

# ===== ONGLET 5 : ECONOMETRIE EXPLIQUEE =====
with tab5:
    st.subheader("Analyse econometrique : le sentiment predit-il le cours ?")

    # =======================================================
    # GLOSSAIRE PEDAGOGIQUE
    # =======================================================
    with st.expander("📚 Glossaire : comprendre les chiffres (cliquez pour ouvrir)", expanded=False):
        st.markdown("""
### 🎯 Qu'est-ce qu'on cherche ?

On veut savoir si **le sentiment des news** (positif/négatif) **influence le cours de bourse** des banques tunisiennes.

Pour cela, on fait une **régression linéaire** : on trace la "meilleure droite" qui relie le sentiment au rendement futur du cours.

---

### 📊 Les 4 chiffres clés

| Indicateur | Signification | Valeurs | Comment lire |
|---|---|---|---|
| **Coefficient** | Combien le cours varie quand le sentiment augmente de +1.0 | +0.088 | **+0.088** = +8.8% de rendement à J+20 |
| **R²** | Qualité du modèle (0 à 1) | 0.025 | **2.5%** de la variation est expliquée. En finance, >2% est **déjà bon** |
| **P-value** | Probabilité que ce soit un hasard | 0.0067 | **< 0.05** = ✅ significatif. **< 0.01** = ✅✅ très significatif |
| **n** | Nombre d'observations | 292 | Plus c'est grand, plus c'est fiable |

---

### 🎯 Les horizons temporels

- **J+1** : le lendemain de la news → effet immédiat
- **J+5** : 5 jours après → effet à court terme
- **J+20** : 20 jours après (≈ 1 mois) → effet à moyen terme

**Pourquoi 3 horizons ?** Pour voir si l'effet est immédiat, progressif, ou différé.

---

### 🚦 Comment interpréter

| Résultat | Signification |
|---|---|
| **p < 0.05** ✅ | L'effet est **réel**, pas un hasard |
| **p > 0.05** ❌ | On **ne peut pas conclure** (effet peut-être nul) |
| **Coefficient > 0** | Le sentiment **monte** → le cours **monte** |
| **Coefficient < 0** | Le sentiment **monte** → le cours **baisse** (rare) |
| **R² élevé** | Le sentiment explique bien le cours |
| **R² faible** | Le sentiment explique peu (mais peut rester utile) |

---

### ⚠️ Important : corrélation ≠ causalité

Si p < 0.05, cela veut dire qu'il y a **une relation statistique**. Pas forcément que **A cause B**. Cela peut être :
- Le sentiment cause le cours (hypothèse)
- Le cours cause le sentiment (feedback)
- Les deux sont causés par un 3ème facteur (économie, taux...)
        """)

    # =======================================================
    # RESULTATS
    # =======================================================
    if not REG_OK:
        st.error(f"Module regression indisponible : {REG_ERROR}")
    else:
        with st.spinner("Calcul de la regression..."):
            reg_df, reg_output = load_regression()

        if reg_df is None:
            st.warning("Pas assez de donnees pour la regression.")
        else:
            results, events = reg_output
            st.success(f"✅ Régression calculée sur **{len(reg_df)} observations** (news croisées avec les prix)")

            # Cartes par horizon avec explication
            st.markdown("### 🎯 Résultats par horizon")

            for horizon, r in results.items():
                if "error" in r:
                    st.error(f"**{horizon}** : {r['error']}")
                    continue

                sig = r["significant"]
                color = "#27AE60" if sig else "#95A5A6"
                bg = "#27AE6022" if sig else "#95A5A622"
                sig_txt = "✅ SIGNIFICATIF" if sig else "❌ Non significatif"

                with st.expander(
                    f"{horizon} — Coefficient : {r['slope']:+.4f} — p-value : {r['p_value']:.4f} — {sig_txt}",
                    expanded=sig
                ):
                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Coefficient",
                        f"{r['slope']:+.4f}",
                        help="Variation du rendement quand le sentiment augmente de +1.0"
                    )
                    col2.metric(
                        "R²",
                        f"{r['r_squared']:.4f}",
                        help="% de la variation expliqué par le modèle (0 à 1)"
                    )
                    col3.metric(
                        "P-value",
                        f"{r['p_value']:.4f}",
                        help="Probabilité que ce soit un hasard. < 0.05 = significatif"
                    )
                    col4.metric(
                        "Observations",
                        r["n"],
                        help="Nombre de news analysées"
                    )

                    # Explication en langage naturel
                    if sig:
                        direction = "augmente" if r["slope"] > 0 else "baisse"
                        pct = r["slope"] * 100
                        st.success(
                            f"**📖 Lecture :** Quand le sentiment d'une news augmente de **+1.0**, "
                            f"le cours de la banque **{direction}** en moyenne de **{pct:+.2f}%** "
                            f"sur **{horizon}**. Cette relation est **fiable** "
                            f"(seulement {r['p_value']*100:.2f}% de chance que ce soit un hasard)."
                        )
                    else:
                        st.info(
                            f"**📖 Lecture :** Aucune relation claire détectée à **{horizon}**. "
                            f"Le coefficient est de {r['slope']:+.4f}, mais la p-value de "
                            f"{r['p_value']:.4f} est **trop élevée** (> 0.05) → on ne peut pas "
                            f"conclure que le sentiment influence le cours à cet horizon."
                        )

                    # Petite explication R²
                    if r["r_squared"] < 0.01:
                        r2_txt = "très faible (le sentiment explique peu le cours)"
                    elif r["r_squared"] < 0.05:
                        r2_txt = "faible mais normal en finance"
                    elif r["r_squared"] < 0.15:
                        r2_txt = "bon pour de la finance"
                    else:
                        r2_txt = "excellent (rare en finance)"

                    st.caption(f"💡 **R² = {r['r_squared']:.4f}** signifie que **{r['r_squared']*100:.2f}%** "
                               f"de la variation du cours est expliquée par le sentiment → {r2_txt}.")

            # =======================================================
            # SYNTHESE GLOBALE
            # =======================================================
            st.markdown("---")
            st.markdown("### 📝 Synthèse globale")

            # Compter les significatifs
            sig_horizons = [h for h, r in results.items() if "error" not in r and r["significant"]]

            if sig_horizons:
                best_h = max(sig_horizons, key=lambda h: abs(results[h]["slope"]))
                best = results[best_h]
                direction = "haussier" if best["slope"] > 0 else "baissier"
                st.success(
                    f"### ✅ Résultat principal\n\n"
                    f"Le sentiment des news **prédit significativement** le cours à **{best_h}**.\n\n"
                    f"- **Effet** : {best['slope']*100:+.2f}% par unité de sentiment\n"
                    f"- **Sens** : {direction}\n"
                    f"- **Fiabilité** : p-value = {best['p_value']:.4f} ({'très fiable' if best['p_value'] < 0.01 else 'fiable'})\n"
                    f"- **Échantillon** : {best['n']} observations\n\n"
                    f"👉 **En clair** : quand une banque reçoit des news positives, "
                    f"son cours a tendance à **{'monter' if best['slope'] > 0 else 'baisser'}** "
                    f"sur un horizon de **{best_h}**."
                )
            else:
                st.warning(
                    "### ⚠️ Aucune relation significative détectée\n\n"
                    "Sur cette période, le sentiment des news ne prédit pas clairement "
                    "le cours des banques tunisiennes. C'est un résultat **honnête** : "
                    "en finance, beaucoup de signaux ne sont pas statistiquement significatifs."
                )

            # =======================================================
            # NUAGE DE POINTS
            # =======================================================
            st.markdown("---")
            st.markdown("### 🔬 Visualisation : sentiment vs rendement")

            st.markdown("""
**Comment lire ce graphique :**
- Chaque **point bleu** = une news
- **Axe X** = sentiment de la news (négatif à gauche, positif à droite)
- **Axe Y** = rendement du cours (en %) après la news
- **Droite rouge** = la "meilleure droite" qui relie les deux (régression)
- **Si la droite monte** → sentiment positif = rendement positif (✅ signal)
- **Si la droite est plate** → aucun lien entre les deux
            """)

            horizon_choice = st.selectbox("Horizon à visualiser", ["J+1", "J+5", "J+20"], index=2)
            h = int(horizon_choice.replace("J+", ""))
            target = f"ret_{h}"

            sub_plot = reg_df.dropna(subset=[target, "sentiment"])
            if len(sub_plot) > 0:
                x = sub_plot["sentiment"].values
                y = sub_plot[target].values * 100

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers",
                    marker=dict(size=6, color="#2E86DE", opacity=0.5),
                    name="News",
                    hovertemplate="Sentiment : %{x:+.2f}<br>Rendement : %{y:+.2f}%<extra></extra>",
                ))

                if len(x) > 1:
                    slope, intercept, r_val, p_val, _ = stats.linregress(x, sub_plot[target].values)
                    x_line = np.linspace(x.min(), x.max(), 50)
                    y_line = (slope * x_line + intercept) * 100
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode="lines",
                        line=dict(color="#E74C3C", width=3),
                        name=f"Régression (p={p_val:.4f})",
                    ))

                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.add_vline(x=0, line_dash="dot", line_color="gray")
                fig.update_layout(
                    height=500,
                    xaxis_title="Sentiment de la news",
                    yaxis_title=f"Rendement du cours {horizon} (%)",
                    hovermode="closest",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats résumées
                r_simple = sub_plot["sentiment"].corr(sub_plot[target])
                st.caption(
                    f"**Corrélation de Pearson** : {r_simple:+.4f} — "
                    f"interprétation : {'positive (sentiment ↑ → cours ↑)' if r_simple > 0.05 else ('négative' if r_simple < -0.05 else 'proche de zéro (pas de lien linéaire)')}"
                )

            # =======================================================
            # ANALYSE PAR EVENEMENT
            # =======================================================
            st.markdown("---")
            st.markdown("### 📊 Analyse par type d'événement")

            st.markdown("""
**Question** : est-ce que certains types de news ont **plus d'impact** que d'autres ?

On a classé les news en catégories :
- 📊 **resultats** : annonces de bénéfices, PNB, chiffre d'affaires
- 💰 **dividende** : distributions aux actionnaires
- ⚖️ **regulation** : décisions de la BCT, amendes, lois
- 🌍 **macro** : inflation, taux, économie générale
- 🤝 **fusion** : acquisitions, partenariats
- 👔 **direction** : nominations, départs de dirigeants
            """)

            if len(events) > 0:
                # Tableau enrichi
                events_display = events.copy()
                events_display["sentiment_moyen"] = events_display["sentiment_moyen"].round(3)
                events_display["ret_1_moyen"] = (events_display["ret_1_moyen"] * 100).round(2)
                events_display["ret_5_moyen"] = (events_display["ret_5_moyen"] * 100).round(2)
                events_display["ret_20_moyen"] = (events_display["ret_20_moyen"] * 100).round(2)
                events_display.columns = ["Événement", "Nb news", "Sentiment moyen",
                                          "Rendement J+1 (%)", "Rendement J+5 (%)", "Rendement J+20 (%)"]

                st.dataframe(events_display, use_container_width=True)

                # Graphique
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=events["event"],
                    y=events["ret_20_moyen"] * 100,
                    marker_color=["#27AE60" if v > 0 else "#E74C3C" for v in events["ret_20_moyen"]],
                    text=[f"{v*100:+.2f}%" for v in events["ret_20_moyen"]],
                    textposition="outside",
                ))
                fig2.add_hline(y=0, line_color="black", line_width=1)
                fig2.update_layout(
                    height=400,
                    yaxis_title="Rendement moyen J+20 (%)",
                    xaxis_title="Type d'événement",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Insight
                if len(events) > 0:
                    top_event = events.iloc[0]
                    st.info(
                        f"💡 **Insight** : le type d'événement avec le **plus fort impact positif** "
                        f"est **{top_event['event']}** (rendement moyen {top_event['ret_20_moyen']*100:+.2f}% à J+20 "
                        f"sur {int(top_event['n'])} news)."
                    )

            st.caption(
                "⚠️ Analyse statistique à but pédagogique. Corrélation n'implique pas causalité. "
                "Ne constitue pas un conseil en investissement."
            )

# ===== ONGLET 6 =====
with tab6:
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

        if st.button("Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

st.warning("Outil pedagogique. Pas un conseil en investissement.")