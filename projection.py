"""Projection récursive sans observation postérieure à la date d'origine."""
import numpy as np
import pandas as pd
from train import features, NUMS, CATS


def project(history, model, origin, months=6):
    origin = pd.Timestamp(origin)
    known = history[history.Date <= origin].copy()
    current = known[known.Date == origin].copy()
    if current.empty:
        raise ValueError("Aucune observation à la date d'origine.")
    source = current.set_index('IdCodeEAN').UN_SOG
    outputs = []
    for horizon in range(1, months + 1):
        step_origin = origin + pd.DateOffset(months=horizon - 1)
        target = origin + pd.DateOffset(months=horizon)
        inputs = features(known)
        inputs = inputs[inputs.Date == step_origin].copy()
        pred = np.maximum(0, np.expm1(np.clip(inputs.log_volume + model['reg'].predict(inputs[NUMS+CATS]), 0, 25)))
        result = inputs[['IdCodeEAN','NomPresentation','Laboratoire','Marque','Segment','SousSegment']].copy()
        result['Date'] = origin
        result['Date_cible'] = target
        result['Horizon'] = horizon
        result['UN_SOG'] = result.IdCodeEAN.map(source)
        result['prediction_ia'] = pred
        result['prediction_reference'] = result.UN_SOG
        seasonal = known[known.Date == target - pd.DateOffset(years=1)].set_index('IdCodeEAN').UN_SOG
        result['prediction_saisonniere'] = result.IdCodeEAN.map(seasonal).fillna(result.UN_SOG)
        result['proba_baisse'] = np.nan
        if horizon == 1:
            positive = inputs.UN_SOG > 0
            if positive.any():
                result.loc[positive, 'proba_baisse'] = model['clf'].predict_proba(inputs.loc[positive, NUMS+CATS])[:, 1]
        result['alerte_classification'] = result.proba_baisse >= model['threshold']
        result['alerte'] = result.alerte_classification if horizon == 1 else (result.UN_SOG.gt(0) & result.prediction_ia.le(.8*result.UN_SOG))
        result['ventes_futures'] = np.nan
        outputs.append(result)
        # Distribution, VMM et stocks gardés à leur dernière valeur connue.
        # Seules les ventes et les dates sont projetées ; aucune vérité future jointe.
        next_month = inputs[history.columns].copy()
        next_month['Date'] = target
        next_month['Annee'] = target.year
        next_month['Mois'] = target.month
        next_month['UN_SOG'] = pred
        known = pd.concat([known, next_month], ignore_index=True)
    return pd.concat(outputs, ignore_index=True)
