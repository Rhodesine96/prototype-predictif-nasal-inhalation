import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Application de démonstration — Prototype prédictif complémentaire à C-Visual
# Laboratoire GERS DATA — Marché Nasal Inhalation (MAYOLY SPINDLER)
#
# Développée par Rhodesine DONLEFACK dans le cadre de sa thèse professionnelle
# (Nexa Digital School — Mastère Data et Intelligence Artificielle).
#
# Identité visuelle inspirée de la charte GERS DATA / C-Visual : bleu marine,
# teal et orange, jauges semi-circulaires, sidebar sombre.
#
# RGPD : aucune donnée personnelle ou de patient n'est utilisée. Les données
# sous-jacentes sont des ventes agrégées par produit/mois, issues du marché
# SOG Conseil / INDIC (GERS DATA), enrichies de données de linéaire (C-Visual).
# Aucune saisie utilisateur n'est stockée : les calculs sont réalisés en
# mémoire, à la volée, pour la durée de la session.
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Palette GERS DATA / C-Visual ---
NAVY = "#1B2A4A"
NAVY_DARK = "#12203A"
TEAL = "#2E8B7A"
TEAL_LIGHT = "#4FB3A0"
ORANGE = "#D9822B"
ORANGE_LIGHT = "#F0A94E"
GREY_BG = "#F4F6FA"
RED = "#B33951"

st.set_page_config(
    page_title="C-Visual Prédictif | GERS DATA",
    page_icon="💊",
    layout="wide",
)

def get_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

logo_b64 = get_base64(os.path.join(BASE_DIR, "gers_style_icon.png"))

st.markdown(f"""
<style>
    .stApp {{ background-color: {GREY_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #E8EDF5 !important; }}
    h1, h2, h3 {{ color: {NAVY} !important; font-weight: 700 !important; }}
    .header-bar {{
        background: linear-gradient(90deg, {NAVY} 0%, {NAVY_DARK} 100%);
        padding: 18px 28px; border-radius: 10px; margin-bottom: 22px;
        display: flex; align-items: center; box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }}
    .header-title {{ color: white; font-size: 26px; font-weight: 800; margin: 0; letter-spacing: 0.5px; }}
    .header-title span.accent {{ color: {ORANGE_LIGHT}; }}
    .header-subtitle {{ color: #AEB9CC; font-size: 13px; margin: 2px 0 0 0; }}
    .kpi-card {{
        background: white; border-radius: 12px; padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(20,30,60,0.08); border-top: 4px solid {TEAL};
        text-align: center; height: 100%;
    }}
    .kpi-card.orange {{ border-top-color: {ORANGE}; }}
    .kpi-label {{ color: #6B7A99; font-size: 12.5px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; margin-bottom: 4px; }}
    .kpi-value {{ color: {NAVY}; font-size: 30px; font-weight: 800; }}
    .badge {{ display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 15px; color: white; }}
    .stButton>button {{ background-color: {ORANGE}; color: white; font-weight: 700; border-radius: 8px; border: none; padding: 0.6em 1.4em; }}
    .stButton>button:hover {{ background-color: {ORANGE_LIGHT}; color: white; }}
    div[data-testid="stForm"] {{ background: white; padding: 22px; border-radius: 12px; box-shadow: 0 2px 8px rgba(20,30,60,0.06); }}
</style>
""", unsafe_allow_html=True)

logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:54px;margin-right:18px;">' if logo_b64 else ""
st.markdown(f"""
<div class="header-bar">
    {logo_html}
    <div>
        <p class="header-title">C-VISUAL <span class="accent">PRÉDICTIF</span></p>
        <p class="header-subtitle">by GERS DATA · Marché Nasal Inhalation · Mayoly Spindler</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "⚠️ Prototype de démonstration développé dans le cadre d'une thèse professionnelle. "
    "Il ne remplace pas les rapports officiels C-Visual / OPTIPHARMA et n'est pas déployé "
    "en production chez GERS DATA.",
    icon="ℹ️",
)

with st.sidebar:
    if logo_b64:
        st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="height:60px;">', unsafe_allow_html=True)
    st.markdown("### Pages")
    st.markdown("**▸ Simulation**")
    st.markdown("Accueil (démonstration)")
    st.divider()
    st.markdown("### Accessibilité")
    taille_police = st.select_slider("Taille du texte", options=["Standard", "Grand", "Très grand"], value="Standard")
    st.divider()
    st.caption("Prototype Streamlit — Python\nDonnées GERS DATA (SOG Conseil / INDIC)\nEnrichies C-Visual (linéaire)\nConforme RGPD")

taille_px = {"Standard": 16, "Grand": 19, "Très grand": 23}[taille_police]
st.markdown(f"<style>html, body, [class*='css'] {{ font-size: {taille_px}px; }}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(BASE_DIR, "model_regression.pkl"))
    clf = joblib.load(os.path.join(BASE_DIR, "model_classification.pkl"))
    meta = joblib.load(os.path.join(BASE_DIR, "meta.pkl"))
    return reg, clf, meta

try:
    model_reg, model_clf, meta = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    st.error(f"Impossible de charger les modèles : {e}")

def gauge(value, max_value, title, color, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': suffix, 'font': {'size': 26, 'color': NAVY}},
        title={'text': title, 'font': {'size': 13, 'color': "#6B7A99"}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 0, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': "white", 'borderwidth': 0,
            'steps': [{'range': [0, max_value], 'color': "#EDF0F6"}],
        }
    ))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=40, b=5), paper_bgcolor="rgba(0,0,0,0)")
    return fig

if models_ok:
    MOIS_LABELS = ["Janvier","Février","Mars","Avril","Mai","Juin",
                   "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

    st.markdown("## 1. Caractéristiques de la référence à évaluer")

    with st.form("formulaire_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            laboratoire = st.selectbox("Laboratoire", options=meta["laboratoires"],
                index=meta["laboratoires"].index("MAYOLY-SPINDLER") if "MAYOLY-SPINDLER" in meta["laboratoires"] else 0)
            marque = st.selectbox("Marque", options=meta["marques"])
            segment = st.selectbox("Segment (macro-forme)", options=meta["segments"])
            sous_segment = st.selectbox("Sous-segment (forme galénique)", options=meta["sous_segments"])
            mois = st.selectbox("Mois de la campagne évaluée", options=list(range(1, 13)), format_func=lambda m: MOIS_LABELS[m-1])
        with col2:
            prix = st.number_input("Prix TTC médian (€)", min_value=0.0, value=6.50, step=0.10)
            vmm = st.number_input("Vente Moyenne Mensuelle par pharmacie détentrice (VMM)", min_value=0.0, value=2.0, step=0.1)
            smm = st.number_input("Stock Moyen Mensuel (SMM)", min_value=0.0, value=3.0, step=0.1)
            dnvstandard = st.slider("Distribution Numérique Valeur (DNVSTANDARD, %)", 0.0, 100.0, 45.0)
            dvv = st.slider("Distribution Valeur (DVV, %)", 0.0, 100.0, 40.0)
            dns = st.slider("Distribution Numérique (DNS, %)", 0.0, 100.0, 40.0)
            dvs = st.slider("Distribution Valeur Standard (DVS, %)", 0.0, 100.0, 35.0)

        st.markdown("**Présence en linéaire (facultatif, laisser à 0 si inconnu)**")
        col3, col4 = st.columns(2)
        with col3:
            facing = st.number_input("Nombre total de facings moyens (panel)", min_value=0.0, value=0.0, step=1.0)
        with col4:
            etageres = st.number_input("Nombre d'étagères occupées", min_value=0, value=0, step=1)

        submitted = st.form_submit_button("🔍 Lancer la prédiction")

    if submitted:
        labo_freq = meta["freq_labo"].get(laboratoire, 0.0)
        marque_freq = meta["freq_marque"].get(marque, 0.0)
        mois_sin = np.sin(2 * np.pi * mois / 12)
        mois_cos = np.cos(2 * np.pi * mois / 12)

        X_reg = pd.DataFrame([{
            "DNVSTANDARD": dnvstandard, "DVV": dvv, "DNS": dns, "DVS": dvs,
            "prixTTC_median": prix, "VMM": vmm, "SMM": smm,
            "Labo_freq": labo_freq, "Marque_freq": marque_freq,
            "Mois_sin": mois_sin, "Mois_cos": mois_cos,
            "Facing_total_moyen": facing, "Nb_etageres_occupees": etageres,
            "Segment": segment, "SousSegment": sous_segment,
        }])
        pred_log = model_reg.predict(X_reg)[0]
        pred_un = max(0, np.expm1(pred_log))

        X_clf = pd.DataFrame([{
            "prixTTC_median": prix, "VMM": vmm, "SMM": smm,
            "Labo_freq": labo_freq, "Marque_freq": marque_freq,
            "Mois_sin": mois_sin, "Mois_cos": mois_cos,
            "Segment": segment, "SousSegment": sous_segment,
        }])
        pred_classe = model_clf.predict(X_clf)[0]
        proba = model_clf.predict_proba(X_clf)[0]
        classes = model_clf.named_steps["model"].classes_
        proba_dict = dict(zip(classes, proba))
        confiance = max(proba) * 100

        class_colors = {"Locomotive": TEAL, "Pepite a developper": ORANGE, "Sous-performant": "#C9A227", "Marginal": RED}
        badge_color = class_colors.get(pred_classe, NAVY)

        st.markdown("## 2. Résultats de la prédiction")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Volume mensuel estimé</div>
                <div class="kpi-value">{pred_un:,.0f}</div><div class="kpi-label">unités / mois</div></div>""".replace(",", " "), unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card orange"><div class="kpi-label">Classe de performance</div><br>
                <span class="badge" style="background-color:{badge_color};">{pred_classe}</span></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Confiance du modèle</div>
                <div class="kpi-value">{confiance:.1f}%</div><div class="kpi-label">probabilité classe prédite</div></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card orange"><div class="kpi-label">Segment</div>
                <div class="kpi-value" style="font-size:18px;">{segment}</div><div class="kpi-label">{sous_segment[:28]}</div></div>""", unsafe_allow_html=True)

        st.write("")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge(dnvstandard, 100, "Distribution (DNVSTANDARD)", TEAL, "%"), use_container_width=True)
        with g2:
            st.plotly_chart(gauge(min(confiance, 100), 100, "Confiance du modèle", ORANGE, "%"), use_container_width=True)
        with g3:
            st.plotly_chart(gauge(min(pred_un, 5000), 5000, "Volume estimé (plafonné à 5000)", TEAL_LIGHT, ""), use_container_width=True)

        st.markdown("### Répartition des probabilités par classe")
        d1, d2 = st.columns([1, 1.3])
        with d1:
            colors_pie = [class_colors.get(c, NAVY) for c in proba_dict.keys()]
            fig_donut = go.Figure(data=[go.Pie(
                labels=list(proba_dict.keys()), values=list(proba_dict.values()),
                hole=0.55, marker=dict(colors=colors_pie, line=dict(color='white', width=2)),
                textinfo='percent', textfont=dict(size=13))])
            fig_donut.update_layout(showlegend=True, height=300, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.15),
                annotations=[dict(text=pred_classe.split()[0], x=0.5, y=0.5, font_size=14, showarrow=False, font_color=NAVY)])
            st.plotly_chart(fig_donut, use_container_width=True)
        with d2:
            libelles = {
                "Locomotive": "Fort volume ET forte distribution attendus : référence motrice sur son marché.",
                "Pepite a developper": "Fort volume mais distribution encore faible : potentiel sous-exploité, à pousser en référencement.",
                "Sous-performant": "Distribution correcte mais volume attendu faible : présence en pharmacie non valorisée commercialement.",
                "Marginal": "Volume et distribution attendus faibles : référence de niche ou en difficulté.",
            }
            st.markdown(f"**Lecture métier :** {libelles.get(pred_classe, '')}")
            proba_df = pd.DataFrame({"Classe": list(proba_dict.keys()), "Probabilité": list(proba_dict.values())}).sort_values("Probabilité", ascending=False)
            st.dataframe(proba_df.style.format({"Probabilité": "{:.1%}"}), hide_index=True, use_container_width=True)

        st.caption(
            "Rappel méthodologique : les variables ayant servi à construire la classe de "
            "performance (volume de vente, distribution) sont exclues des variables "
            "prédictives de la classification, afin d'éviter toute fuite de données et de "
            "tester une réelle capacité anticipative."
        )
else:
    st.warning("Les modèles n'ont pas pu être chargés. Vérifiez la présence des fichiers "
               "model_regression.pkl, model_classification.pkl et meta.pkl dans le même dossier que app.py.")

st.divider()
st.caption(
    "Prototype développé avec Streamlit — Python. Données : GERS DATA (SOG Conseil / INDIC), "
    "marché Nasal Inhalation, enrichies de données de linéaire C-Visual. "
    "Aucune donnée personnelle traitée, conforme RGPD."
)
