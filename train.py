"""Expérience nouvelle du 8 octobre 2026 : t -> t+1, validation et test chronologiques."""
from pathlib import Path
import json,time,sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,fbeta_score,average_precision_score,roc_auc_score,confusion_matrix
ROOT=Path(__file__).resolve().parent
CATS=['Laboratoire','Marque','Segment','SousSegment']
NUMS=['log_volume','log_lag1','log_lag2','log_lag3','log_lag6','log_lag11','log_lag12','log_moyenne3','variation1','DNVSTANDARD','DVV','DNS','DVS','VMM','SMM','mois_cible_sin','mois_cible_cos','temps']

def read():
 with sqlite3.connect(f'file:{ROOT / "nasal_inhalation.db"}?mode=ro',uri=True) as c:d=pd.read_sql('select * from ventes',c)
 d['Date']=pd.to_datetime(dict(year=d.Annee,month=d.Mois,day=1))
 assert not d.duplicated(['IdCodeEAN','Date']).any()
 return d.sort_values(['IdCodeEAN','Date'])

def features(d):
 f=d.copy();f['log_volume']=np.log1p(f.UN_SOG.clip(lower=0))
 for k in [1,2,3,6,11,12]:
  lookup=d[['IdCodeEAN','Date','UN_SOG']].copy();lookup.Date+=pd.DateOffset(months=k)
  lookup=lookup.rename(columns={'UN_SOG':f'lag{k}'})
  f=f.merge(lookup,on=['IdCodeEAN','Date'],how='left',validate='one_to_one')
  f[f'log_lag{k}']=np.log1p(f[f'lag{k}'].clip(lower=0))
 f['log_moyenne3']=np.log1p(f[['UN_SOG','lag1','lag2']].mean(axis=1))
 f['variation1']=(f.log_volume-f.log_lag1).fillna(0)
 f['Date_cible']=f.Date+pd.DateOffset(months=1)
 f['mois_cible_sin']=np.sin(2*np.pi*f.Date_cible.dt.month/12);f['mois_cible_cos']=np.cos(2*np.pi*f.Date_cible.dt.month/12)
 f['temps']=(f.Date.dt.year-2023)*12+f.Date.dt.month-7
 # No facings or price: dates/initial price imputations do not support availability at forecast origin.
 target=d[['IdCodeEAN','Date','UN_SOG']].rename(columns={'Date':'Date_cible','UN_SOG':'ventes_futures'})
 f=f.merge(target,on=['IdCodeEAN','Date_cible'],how='left',validate='one_to_one')
 return f

def preprocess(scale=False,cats=True):
 num=[('impute',SimpleImputer(strategy='median',add_indicator=True))]
 if scale:num.append(('scale',StandardScaler()))
 return ColumnTransformer([('num',Pipeline(num),NUMS)]+([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATS)] if cats else []))

def reg_metrics(y,p):
 p=np.maximum(0,p)
 return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MSE':float(mean_squared_error(y,p)),'R2_unites':float(r2_score(y,p)),'WAPE':float(np.abs(y-p).sum()/np.sum(y)),'MAE_log':float(mean_absolute_error(np.log1p(y),np.log1p(p)))}

def cls_metrics(y,p,threshold):
 q=np.asarray(p)>=threshold
 return {'Accuracy':float(accuracy_score(y,q)),'Balanced_accuracy':float(balanced_accuracy_score(y,q)),'Precision':float(precision_score(y,q,zero_division=0)),'Recall':float(recall_score(y,q,zero_division=0)),'F2':float(fbeta_score(y,q,beta=2,zero_division=0)),'AP':float(average_precision_score(y,p)),'AUC':float(roc_auc_score(y,p)),'Confusion':confusion_matrix(y,q).tolist(),'Threshold':float(threshold),'Alerts':int(q.sum())}

def main():
 t0=time.perf_counter();d=read();f=features(d);pairs=f[f.ventes_futures.notna()].copy()
 train=pairs[pairs.Date_cible<='2025-10-01'];test=pairs[pairs.Date_cible>='2025-11-01']
 inner=train[train.Date_cible<='2025-04-01'];val=train[train.Date_cible>='2025-05-01']
 report={'date':'2026-10-08','rows_source':len(d),'products':int(d.IdCodeEAN.nunique()),'pairs':len(pairs),'train':len(train),'train_pct':len(train)/len(pairs),'validation':len(val),'inner_train':len(inner),'test':len(test),'train_targets':['2023-08','2025-10'],'validation_targets':['2025-05','2025-10'],'test_targets':['2025-11','2026-06'],'reg_validation':{},'reg_test':{},'risk_validation':{},'risk_test':{},'threshold_policy':'Maximiser F2 sur validation sous précision >= 50% et <= 50% de références signalées ; repli au F2 si aucun seuil admissible.'}
 y=np.log1p(inner.ventes_futures)-inner.log_volume
 candidates={
 'Ridge_residuel_alpha1':Pipeline([('preprocess',preprocess(True)),('model',Ridge(alpha=1))]),
 'Ridge_residuel_alpha100':Pipeline([('preprocess',preprocess(True)),('model',Ridge(alpha=100))]),
 'Foret_residuelle_150':Pipeline([('preprocess',preprocess()),('model',RandomForestRegressor(n_estimators=150,max_depth=12,min_samples_leaf=8,n_jobs=2,random_state=42))]),
 'Boosting_residuel':Pipeline([('preprocess',preprocess(cats=False)),('model',HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,l2_regularization=5,learning_rate=.05,random_state=42))])}
 report['reg_validation']['Persistance']=reg_metrics(val.ventes_futures,val.UN_SOG)
 for name,model in candidates.items():
  model.fit(inner[NUMS+CATS],y)
  pred=np.expm1(np.clip(val.log_volume+model.predict(val[NUMS+CATS]),0,25))
  report['reg_validation'][name]=reg_metrics(val.ventes_futures,pred)
  print('REG VAL',name,report['reg_validation'][name],flush=True)
 best=min(candidates,key=lambda name:report['reg_validation'][name]['MAE'])
 validation_reg_pred=np.maximum(0,np.expm1(np.clip(val.log_volume+candidates[best].predict(val[NUMS+CATS]),0,25)))
 reg=candidates[best];reg.fit(train[NUMS+CATS],np.log1p(train.ventes_futures)-train.log_volume)
 ptest=np.maximum(0,np.expm1(np.clip(test.log_volume+reg.predict(test[NUMS+CATS]),0,25)))
 report['reg_selected']=best
 report['reg_test'][best]=reg_metrics(test.ventes_futures,ptest)
 report['reg_test']['Persistance']=reg_metrics(test.ventes_futures,test.UN_SOG)
 report['reg_validation']['Saisonnier']=reg_metrics(val.ventes_futures,val.lag11.fillna(val.UN_SOG))
 report['reg_test']['Saisonnier']=reg_metrics(test.ventes_futures,test.lag11.fillna(test.UN_SOG))
 # Binary task: at least a 20% fall in monthly sales, conditional on a positive origin volume.
 ri=inner[inner.UN_SOG>0];rv=val[val.UN_SOG>0];rt=train[train.UN_SOG>0];re=test[test.UN_SOG>0]
 yi=(ri.ventes_futures<=.8*ri.UN_SOG).astype(int);yv=(rv.ventes_futures<=.8*rv.UN_SOG).astype(int);yt=(rt.ventes_futures<=.8*rt.UN_SOG).astype(int);ye=(re.ventes_futures<=.8*re.UN_SOG).astype(int)
 report['risk_validation']['Regression_seuil20']=cls_metrics(yv,(validation_reg_pred[val.UN_SOG>0]<=.8*rv.UN_SOG).astype(float),.5)
 report['risk_counts']={'inner':len(ri),'validation':len(rv),'train':len(rt),'test':len(re),'validation_events':int(yv.sum()),'test_events':int(ye.sum()),'test_event_rate':float(ye.mean())}
 clfs={
 'Logistique':Pipeline([('preprocess',preprocess(True)),('model',LogisticRegression(max_iter=1200,C=.1,class_weight='balanced',random_state=42))]),
 'Foret_risque':Pipeline([('preprocess',preprocess()),('model',RandomForestClassifier(n_estimators=180,max_depth=10,min_samples_leaf=10,class_weight='balanced',n_jobs=2,random_state=42))]),
 'Boosting_risque':Pipeline([('preprocess',preprocess(cats=False)),('model',HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,l2_regularization=10,learning_rate=.05,random_state=42))])}
 thresholds={}
 for name,model in clfs.items():
  model.fit(ri[NUMS+CATS],yi);prob=model.predict_proba(rv[NUMS+CATS])[:,1]
  candidates_threshold=np.arange(.1,.81,.05)
  eligible=[th for th in candidates_threshold if precision_score(yv,prob>=th,zero_division=0)>=.5 and (prob>=th).mean()<=.5]
  threshold=max(eligible or candidates_threshold,key=lambda th:fbeta_score(yv,prob>=th,beta=2,zero_division=0))
  thresholds[name]=float(threshold)
  report['risk_validation'][name]=cls_metrics(yv,prob,threshold)
  print('RISQUE VAL',name,report['risk_validation'][name],flush=True)
 cbest=max(clfs,key=lambda name:report['risk_validation'][name]['F2'])
 clf=clfs[cbest];clf.fit(rt[NUMS+CATS],yt);probs=clf.predict_proba(re[NUMS+CATS])[:,1];threshold=thresholds[cbest]
 report['risk_selected']=cbest;report['risk_threshold']=threshold
 report['risk_test'][cbest]=cls_metrics(ye,probs,threshold)
 report['risk_test']['Aucune_alerte']=cls_metrics(ye,np.zeros(len(re)),.5)
 rule=(re.UN_SOG<=.8*re.lag1).fillna(False).astype(float)
 report['risk_test']['Baisse_precedente']=cls_metrics(ye,rule,.5)
 forecast_flag=(ptest[test.UN_SOG>0]<=.8*re.UN_SOG).astype(float)
 report['risk_test']['Regression_seuil20']=cls_metrics(ye,forecast_flag,.5)
 seasonal_risk=(re.lag11.fillna(re.UN_SOG)<=.8*re.UN_SOG).astype(float)
 report['risk_test']['Saisonnier_seuil20']=cls_metrics(ye,seasonal_risk,.5)
 testout_policy='Alertes principales selon classificateur choisi sur validation, seuil fixé sur validation ; baisse prévue par régression affichée séparément.'
 report['alert_policy']=testout_policy
 # Retrospective output: fixed model fitted through October 2025 targets. These rows are independent test forecasts.
 testout=test[['IdCodeEAN','NomPresentation','Laboratoire','Marque','Segment','SousSegment','Date','Date_cible','UN_SOG','ventes_futures']].copy()
 testout['prediction_ia']=ptest;testout['prediction_reference']=test.UN_SOG.to_numpy();testout['prediction_saisonniere']=test.lag11.fillna(test.UN_SOG).to_numpy();testout['proba_baisse']=np.nan;testout['baisse_reelle']=False
 testout.loc[test.UN_SOG>0,'proba_baisse']=probs
 testout.loc[test.UN_SOG>0,'baisse_reelle']=ye.to_numpy().astype(bool)
 testout['alerte_classification']=testout.proba_baisse>=threshold
 testout['alerte']=testout['alerte_classification']
 testout.to_csv(ROOT/'previsions_test.csv',index=False)
 # Separate production experiment snapshot refitted on all labelled pairs, not used for independent test scores.
 from sklearn.base import clone
 production_reg=clone(reg).fit(pairs[NUMS+CATS],np.log1p(pairs.ventes_futures)-pairs.log_volume)
 positive=pairs[pairs.UN_SOG>0]
 production_clf=clone(clf).fit(positive[NUMS+CATS],(positive.ventes_futures<=.8*positive.UN_SOG).astype(int))
 joblib.dump({'reg':reg,'clf':clf,'num_features':NUMS,'cat_features':CATS,'threshold':threshold},ROOT/'modeles_test.pkl')
 joblib.dump({'reg':production_reg,'clf':production_clf,'num_features':NUMS,'cat_features':CATS,'threshold':threshold},ROOT/'modeles_projection.pkl')
 report['elapsed_s']=time.perf_counter()-t0
 (ROOT/'resultats.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print('FINAL',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
