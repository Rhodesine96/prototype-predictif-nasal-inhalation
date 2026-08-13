# Prototype prédictif — Marché Nasal Inhalation (GERS DATA)

Application développée dans le cadre de la thèse professionnelle de Rhodesine DONLEFACK
(Nexa Digital School — Mastère Data et Intelligence Artificielle).

## Contenu
- `app.py` : application Streamlit (front + back)
- `model_regression.pkl` : modèle de régression (Ridge) prédisant le volume de vente mensuel (UN)
- `model_classification.pkl` : modèle de classification (Random Forest) prédisant la classe de performance commerciale
- `meta.pkl` : listes des laboratoires/marques/segments et fréquences d'encodage
- `requirements.txt` : dépendances Python

## Installation et lancement en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse http://localhost:8501

## Déploiement (pour obtenir une URL publique à citer dans le mémoire)

Le moyen le plus simple et gratuit est **Streamlit Community Cloud** :
1. Déposer ce dossier dans un dépôt GitHub (public ou privé).
2. Se connecter sur https://share.streamlit.io avec ce dépôt.
3. Streamlit Cloud fournit alors une URL publique du type
   `https://<nom-app>.streamlit.app`, à citer dans le mémoire (point VIII du guide).

## RGPD et accessibilité
Voir les commentaires en en-tête de `app.py` : aucune donnée personnelle traitée,
aucune donnée de session conservée, contrôle de taille de police, labels explicites
sur tous les champs, aucune information codée uniquement par la couleur.
