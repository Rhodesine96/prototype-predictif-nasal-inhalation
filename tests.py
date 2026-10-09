from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent if (Path(__file__).resolve().parent/'app.py').exists() else Path('/workspace/scratch/00ca0cfe42eb/outputs/solution_ia');sys.path.insert(0,str(P))
from streamlit.testing.v1 import AppTest
from train import read,features
import pandas as pd
# Isolation temporelle des entrées après suppression des observations futures.
d=read();cut=pd.Timestamp('2025-10-01');a=features(d);b=features(d[d.Date<=cut]);cols=['IdCodeEAN','Date','log_lag11','log_moyenne3','variation1']
pd.testing.assert_frame_equal(a[a.Date<=cut][cols].reset_index(drop=True),b[cols].reset_index(drop=True))
accounts=json.loads((P/'comptes.json').read_text());results=[]
for name,item in accounts.items():
 at=AppTest.from_file(str(P/'app.py'),default_timeout=40)
 at.session_state['account']={'id':name,'role':item['role'],'laboratoire':item.get('laboratoire')}
 at.run();assert not at.exception,list(at.exception)
 nav=at.radio(key='navigation');pages=nav.options
 if item['role']=='client':
  assert 'Administration' not in pages
 for page in pages:
  at.radio(key='navigation').set_value(page).run();assert not at.exception,(page,list(at.exception))
  if item['role']=='client' and page!='Accueil':
   assert next(x for x in at.selectbox if x.label=='Laboratoire').options==['MAYOLY SPINDLER']
  results.append({'role':item['role'],'page':page,'status':'OK'})
 at.radio(key='usage').set_value('Projection hors échantillon disponible').run();assert not at.exception,list(at.exception)
 results.append({'role':item['role'],'page':'projection','status':'OK'})
(P/'tests_resultats.json').write_text(json.dumps({'date':'2026-10-08','temporal_features':'OK','streamlit':results},ensure_ascii=False,indent=2))
print('OK',len(results),'parcours, isolation temporelle')
