import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------------------------
# Application de démonstration — Prototype prédictif complémentaire à C-Visual
# Laboratoire GERS DATA — Marché Nasal Inhalation (MAYOLY SPINDLER)
#
# Développée par Rhodesine DONLEFACK dans le cadre de sa thèse professionnelle
# (Nexa Digital School — Mastère Data et Intelligence Artificielle).
#
# RGPD : aucune donnée personnelle ou de patient n'est utilisée. Les données
# sous-jacentes sont des ventes agrégées par produit/mois, issues du marché
# SOG Conseil / INDIC (GERS DATA), enrichies de données de linéaire (C-Visual).
# Aucune saisie utilisateur n'est stockée : les calculs sont réalisés en
# mémoire, à la volée, pour la durée de la session.
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Prototype prédictif — Marché Nasal Inhalation",
    page_icon="💊",
    layout="centered",
)

# --- Mesures d'accessibilité ---------------------------------------------
# - Contraste renforcé (thème clair, texte sombre)
# - Taille de police ajustable via un réglage explicite (plutôt que de ne
#   compter que sur le zoom navigateur)
# - Aucune information codée uniquement par la couleur : chaque résultat est
#   toujours accompagné d'un libellé textuel explicite
# - Tous les champs de saisie sont des widgets natifs Streamlit, nativement
#   navigables au clavier et compatibles lecteurs d'écran (labels associés)
with st.sidebar:
    st.header("Accessibilité")
    taille_police = st.select_slider(
        "Taille du texte",
        options=["Standard", "Grand", "Très grand"],
        value="Standard",
        help="Ajuste la taille du texte de l'application pour les personnes malvoyantes.",
    )

taille_px = {"Standard": 16, "Grand": 19, "Très grand": 23}[taille_police]
st.markdown(f"<style>html, body, [class*='css'] {{ font-size: {taille_px}px; }}</style>", unsafe_allow_html=True)

st.title("💊 Prototype prédictif — Marché Nasal Inhalation")
st.caption(
    "Développement complémentaire aux rapports C-Visual et SOG Conseil de GERS DATA. "
    "Cet outil teste une approche prédictive (apprentissage supervisé) là où les rapports "
    "existants restituent une photographie descriptive des campagnes passées."
)

st.info(
    "⚠️ Ceci est un prototype de démonstration développé dans le cadre d'une thèse "
    "professionnelle. Il ne remplace pas les rapports officiels C-Visual / OPTIPHARMA "
    "et n'est pas déployé en production chez GERS DATA.",
    icon="ℹ️",
)

# --- Chargement des modèles et métadonnées --------------------------------
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

if models_ok:
    MOIS_LABELS = ["Janvier","Février","Mars","Avril","Mai","Juin",
                   "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

    st.subheader("1. Caractéristiques de la référence à évaluer")

    with st.form("formulaire_prediction"):
        col1, col2 = st.columns(2)

        with col1:
            laboratoire = st.selectbox(
                "Laboratoire", options=meta["laboratoires"],
                index=meta["laboratoires"].index("MAYOLY-SPINDLER") if "MAYOLY-SPINDLER" in meta["laboratoires"] else 0,
                help="Laboratoire titulaire de la référence."
            )
            marque = st.selectbox(
                "Marque", options=meta["marques"],
                help="Marque commerciale de la référence."
            )
            segment = st.selectbox("Segment (macro-forme)", options=meta["segments"])
            sous_segment = st.selectbox("Sous-segment (forme galénique)", options=meta["sous_segments"])
            mois = st.selectbox("Mois de la campagne évaluée", options=list(range(1, 13)),
                                 format_func=lambda m: MOIS_LABELS[m-1])

        with col2:
            prix = st.number_input("Prix TTC médian (€)", min_value=0.0, value=6.50, step=0.10)
            vmm = st.number_input("Vente Moyenne Mensuelle par pharmacie détentrice (VMM)",
                                   min_value=0.0, value=2.0, step=0.1)
            smm = st.number_input("Stock Moyen Mensuel (SMM)", min_value=0.0, value=3.0, step=0.1)
            dnvstandard = st.slider("Distribution Numérique Valeur (DNVSTANDARD, %)", 0.0, 100.0, 45.0)
            dvv = st.slider("Distribution Valeur (DVV, %)", 0.0, 100.0, 40.0)
            dns = st.slider("Distribution Numérique (DNS, %)", 0.0, 100.0, 40.0)
            dvs = st.slider("Distribution Valeur Standard (DVS, %)", 0.0, 100.0, 35.0)

        st.markdown("**Présence en linéaire (facultatif — laisser à 0 si inconnu)**")
        col3, col4 = st.columns(2)
        with col3:
            facing = st.number_input("Nombre total de facings moyens (panel)", min_value=0.0, value=0.0, step=1.0)
        with col4:
            etageres = st.number_input("Nombre d'étagères occupées", min_value=0, value=0, step=1)

        submitted = st.form_submit_button("Lancer la prédiction")

    if submitted:
        labo_freq = meta["freq_labo"].get(laboratoire, 0.0)
        marque_freq = meta["freq_marque"].get(marque, 0.0)
        mois_sin = np.sin(2 * np.pi * mois / 12)
        mois_cos = np.cos(2 * np.pi * mois / 12)

        # --- Régression : volume de vente prévisionnel ---
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

        # --- Classification : matrice Volume x Distribution ---
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

        st.subheader("2. Résultat de la prédiction")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Volume de vente mensuel estimé (UN)", f"{pred_un:,.0f} unités".replace(",", " "))
        with col_b:
            st.metric("Classe de performance commerciale prévue", pred_classe)

        libelles = {
            "Locomotive": "Fort volume attendu ET forte distribution attendue : référence motrice sur son marché.",
            "Pepite a developper": "Fort volume attendu mais distribution encore faible : potentiel sous-exploité, à pousser en référencement.",
            "Sous-performant": "Distribution correcte mais volume attendu faible : présence en pharmacie non valorisée commercialement.",
            "Marginal": "Volume et distribution attendus faibles : référence de niche ou en difficulté.",
        }
        st.write(f"**Lecture métier :** {libelles.get(pred_classe, '')}")

        st.markdown("**Répartition des probabilités par classe**")
        proba_df = pd.DataFrame({"Classe": list(proba_dict.keys()), "Probabilité": list(proba_dict.values())})
        proba_df = proba_df.sort_values("Probabilité", ascending=False)
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
    "Aucune donnée personnelle traitée — conforme RGPD."
)
