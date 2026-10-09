"""Source CSV fournie -> SQLite, export SQL et benchmark reproductible."""
from pathlib import Path
import sqlite3,time,json,statistics
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
 d=pd.read_csv(ROOT/'dataset_source.csv',dtype={'IdCodeEAN':'int64'})
 with sqlite3.connect(ROOT/'nasal_inhalation.db') as c:
  d.to_sql('ventes',c,if_exists='replace',index=False)
  d.to_sql('ventes_brutes',c,if_exists='replace',index=False)
  c.execute('CREATE INDEX idx_ventes_labo_date ON ventes(Laboratoire,Annee,Mois)')
  c.execute('CREATE UNIQUE INDEX idx_ventes_produit_date ON ventes(IdCodeEAN,Annee,Mois)')
  c.commit()
  result={}
  for table in ['ventes_brutes','ventes']:
   sql=f'SELECT SUM(UN_SOG),COUNT(*) FROM {table} WHERE Laboratoire=? AND Annee=? AND Mois=?'
   params=('MAYOLY SPINDLER',2026,6)
   for _ in range(10):c.execute(sql,params).fetchone()
   elapsed=[]
   for _ in range(200):
    start=time.perf_counter_ns();answer=c.execute(sql,params).fetchone();elapsed.append((time.perf_counter_ns()-start)/1e6)
   result[table]={'mean_ms':statistics.mean(elapsed),'median_ms':statistics.median(elapsed),'repetitions':200,'answer':answer,'query_plan':c.execute('EXPLAIN QUERY PLAN '+sql,params).fetchall()}
  (ROOT/'benchmark_sql.json').write_text(json.dumps(result,indent=2))
  with (ROOT/'export_base.sql').open('w') as f:
   for line in c.iterdump():f.write(line+'\n')
 print(result)
if __name__=='__main__':main()
