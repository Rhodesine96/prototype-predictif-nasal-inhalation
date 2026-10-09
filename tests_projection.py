from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from projection import project
from train import read
from streamlit.testing.v1 import AppTest
P=Path(__file__).resolve().parent
d=read();model=joblib.load(P/'modeles_test.pkl');origin=pd.Timestamp('2025-10-01')
p=project(d,model,origin)
truncated=project(d[d.Date<=origin],model,origin)
pd.testing.assert_frame_equal(p,truncated)
assert sorted(p.Horizon.unique())==list(range(1,7))
assert p.proba_baisse[p.Horizon>1].isna().all()
assert np.isfinite(p.prediction_ia).all() and p.prediction_ia.ge(0).all()
truth=d[['IdCodeEAN','Date','UN_SOG']].rename(columns={'Date':'Date_cible','UN_SOG':'observed'})
j=p.merge(truth,on=['IdCodeEAN','Date_cible'],how='inner');metrics=[]
for horizon,part in j.groupby('Horizon'):
 row={'Horizon':int(horizon),'Mois':part.Date_cible.iloc[0].strftime('%Y-%m'),'Observations':len(part)}
 for method in ['prediction_ia','prediction_reference','prediction_saisonniere']:
  error=part[method]-part.observed
  row[method]={'MAE':float(error.abs().mean()),'RMSE':float(np.sqrt((error**2).mean())),'WAPE':float(error.abs().sum()/part.observed.sum())}
 metrics.append(row)
(P/'evaluation_6mois.json').write_text(json.dumps({'origine':'2025-10','statut':'Test exploratoire récursif à origine fixe ; modèles déjà sélectionnés pendant la reprise, sans nouveau test aveugle ni validation métier.','hypotheses':'Distribution, VMM et stocks constants à la dernière observation connue.','resultats':metrics},ensure_ascii=False,indent=2))
accounts=json.loads((P/'comptes.json').read_text());routes=[]
for name,account in accounts.items():
 at=AppTest.from_file(str(P/'app.py'),default_timeout=90)
 at.session_state['account']={'id':name,'role':account['role'],'laboratoire':account.get('laboratoire')}
 at.run();assert not at.exception,list(at.exception)
 at.radio(key='usage').set_value('Projection hors échantillon disponible').run();assert not at.exception,list(at.exception)
 assert len(at.selectbox(key='projection_month').options)==6
 at.selectbox(key='projection_month').select_index(5).run();assert not at.exception,list(at.exception)
 for page in at.radio(key='navigation').options:
  at.radio(key='navigation').set_value(page).run();assert not at.exception,(page,list(at.exception))
  if page=='Prévision produit':
   assert len(at.get('plotly_chart'))==4
   assert not any('unités' in x.value for x in at.metric)
   charts=at.get('plotly_chart');fig=json.loads(charts[0].proto.spec)
   assert fig['layout']['height']==650
   assert len(fig['data'][1]['x'])==7
  if account['role']=='client' and page!='Accueil':assert at.selectbox(key='lab_Projection hors échantillon disponible2026-06-01 00:00:00').options==['MAYOLY SPINDLER']
  routes.append({'role':account['role'],'page':page,'mois':'2026-12','statut':'OK'})
(P/'tests_6mois.json').write_text(json.dumps({'horizons':6,'invariance_apres_suppression_du_futur':'OK','score_risque_premier_mois_uniquement':'OK','graphiques':4,'parcours':routes},ensure_ascii=False,indent=2))
print('OK : 6 horizons, invariance temporelle, 4 graphiques,',len(routes),'parcours décembre')
print(json.dumps(metrics,ensure_ascii=False))
