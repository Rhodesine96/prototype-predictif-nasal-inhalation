from pathlib import Path
import json,base64,sqlite3,datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from projection import project
from auth import authenticate
from train import read,features,NUMS,CATS
from diagnostic import pdf_report
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Anticipation commerciale Nasal Inhalation',page_icon='💊',layout='wide')
logo=base64.b64encode((ROOT/'gers_style_icon.png').read_bytes()).decode()
st.markdown('''<style>.stApp{background:#F4F6FA;color:#1B2A4A}h1,h2,h3{color:#1B2A4A}.header{background:linear-gradient(90deg,#1B2A4A,#12203A);padding:20px;border-radius:12px;display:flex;align-items:center;gap:24px;margin-bottom:20px}.header img{width:140px}.header h1{font-size:27px;color:white;margin:0}.header p{color:#BDC8DB;margin:6px 0}.stButton button,.stDownloadButton button{background:#D9822B;color:white}div[data-testid="stRadio"]>div[role="radiogroup"]{gap:10px;flex-wrap:wrap}div[data-testid="stRadio"]>div[role="radiogroup"] label{background:white;border:1px solid #D8DDE5;border-radius:8px;padding:9px 12px}[data-testid="stMetric"]{background:white;padding:16px;border-top:4px solid #2E8B7A;border-radius:10px}
section[data-testid="stSidebar"]{background:#1B2A4A;color:#FFFFFF;border-right:1px solid #12203A}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{background:#1B2A4A}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{color:#FFFFFF}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#D6DFEF}
section[data-testid="stSidebar"] div[data-testid="stRadio"]>div[role="radiogroup"] label{background:#263B60;border:1px solid #506586;color:#FFFFFF}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button{color:#FFFFFF}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg{fill:currentColor}
section[data-testid="stSidebar"] .stButton button{background:#D9822B;border-color:#D9822B;color:#FFFFFF}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="header"><img alt="Logo du prototype" src="data:image/png;base64,'+logo+'"><div><h1>ANTICIPATION COMMERCIALE</h1><p>Nasal Inhalation · Projection à six mois et risque de baisse à un mois · Prototype pédagogique</p></div></div>',unsafe_allow_html=True)
account=authenticate(ROOT)

@st.cache_data
def resources():
    d=read();f=features(d);r=json.loads((ROOT/'resultats.json').read_text());test=pd.read_csv(ROOT/'previsions_test.csv',parse_dates=['Date','Date_cible'])
    return d,f,r,test
@st.cache_resource
def artifact(name):return joblib.load(ROOT/name)
data,frame,report,test=resources()
allowed=data if account['role']=='admin' else data[data.Laboratoire==account['laboratoire']]
if allowed.empty:st.error('Aucune donnée pour ce compte.');st.stop()
menu=['Accueil','Prévision produit','Alertes portefeuille','Évaluation IA','Dossier de décision']
if account['role']=='admin':menu.append('Administration')
page=st.radio('Navigation',menu,horizontal=True,label_visibility='collapsed',key='navigation')
mode=st.sidebar.radio('Usage',['Validation historique','Projection hors échantillon disponible'],index=1,key='usage')
if mode=='Validation historique':
    permitted=test if account['role']=='admin' else test[test.Laboratoire==account['laboratoire']]
    dates=sorted(permitted.Date.unique(),reverse=True)
    origin=pd.Timestamp(st.selectbox('Dernier mois connu',dates,format_func=lambda v:pd.Timestamp(v).strftime('%Y-%m'),key='origin'))
    predictions=permitted[permitted.Date==origin].copy()
    target=origin+pd.DateOffset(months=1)
    snapshot=allowed[allowed.Date==origin]
    st.caption('Modèles de test entraînés sur les cibles jusqu’à octobre 2025. Les indicateurs et l’historique affichés pour prévoir sont arrêtés à '+origin.strftime('%Y-%m')+'.')
else:
    origin=allowed.Date.max()
    model=artifact('modeles_projection.pkl')
    @st.cache_data
    def six_months(history, date):
        return project(history, artifact('modeles_projection.pkl'), date, 6)
    projections=six_months(allowed, origin)
    dates=sorted(projections.Date_cible.unique())
    target=pd.Timestamp(st.selectbox('Mois projeté pour les comparaisons et le portefeuille',dates,format_func=lambda v:pd.Timestamp(v).strftime('%Y-%m'),key='projection_month'))
    predictions=projections[projections.Date_cible==target].copy()
    snapshot=allowed[allowed.Date==origin]

if page!='Accueil':
    c1,c2=st.columns(2)
    labs=sorted(predictions.Laboratoire.unique())
    lab=c1.selectbox('Laboratoire',labs,index=labs.index('MAYOLY SPINDLER') if 'MAYOLY SPINDLER' in labs else 0,key='lab_'+mode+str(origin))
    labpred=predictions[predictions.Laboratoire==lab]
    brands=['Toutes les marques']+sorted(labpred.Marque.unique())
    brand=c2.selectbox('Marque',brands,key='brand_'+mode+str(origin)+lab)
    portfolio=labpred if brand=='Toutes les marques' else labpred[labpred.Marque==brand]
    segcol,subcol,productcol=st.columns([1,1,2])
    segments=['Tous les segments']+sorted(portfolio.Segment.unique())
    segment=segcol.selectbox('Segment',segments,key='segment_'+mode+str(origin)+lab+brand)
    if segment!='Tous les segments':portfolio=portfolio[portfolio.Segment==segment]
    subs=['Tous les sous-segments']+sorted(portfolio.SousSegment.unique())
    subsegment=subcol.selectbox('Sous-segment',subs,key='subsegment_'+mode+str(origin)+lab+brand+segment)
    if subsegment!='Tous les sous-segments':portfolio=portfolio[portfolio.SousSegment==subsegment]
    portfolio=portfolio.sort_values('NomPresentation')
    ids=portfolio.IdCodeEAN.tolist();names=dict(zip(portfolio.IdCodeEAN,portfolio.NomPresentation))
    pid=productcol.selectbox('Présentation',ids,format_func=lambda v,names=names:f'{names[v]} — {v}',key='product_'+mode+str(origin)+lab+brand+segment+subsegment)
    selected=portfolio[portfolio.IdCodeEAN==pid].iloc[0]
    st.caption(f'Mois connu : {origin:%Y-%m} → mois prévu : {target:%Y-%m} · {len(portfolio)} présentation(s) dans le portefeuille filtré.')

if page=='Accueil':
    st.subheader('Préparer une revue commerciale avant la prochaine observation')
    st.write('C-Visual présente la performance observée. Ce prototype teste une capacité supplémentaire : projeter les ventes sur six mois à partir du dernier mois connu et signaler les références susceptibles de baisser d’au moins 20 %. Il aide à préparer des vérifications métier, sans décider d’une modification de linéaire.')
    st.markdown('1. Choisir une date de référence et un produit.\n2. Consulter la prévision IA, sa référence simple et le signal de risque.\n3. Examiner les alertes du portefeuille.\n4. Vérifier les scores et les limites.\n5. Exporter un dossier de décision pour discussion.')
    st.info('La prévision du volume et le risque de baisse répondent à deux questions différentes. Une estimation ponctuelle n’est pas une probabilité. Les deux sorties peuvent diverger ; ce désaccord est affiché pour revue.')

elif page=='Prévision produit':
    st.subheader(('Projection de '+origin.strftime('%Y-%m')+' vers les six mois suivants') if mode=='Projection hors échantillon disponible' else ('Prévision pour '+target.strftime('%Y-%m')))
    st.caption('Indicateurs du mois choisi : '+target.strftime('%Y-%m'))
    cols=st.columns(4)
    cols[0].metric('Dernier nombre de ventes connu',f'{selected.UN_SOG:,.0f}')
    cols[1].metric('Prévision IA',f'{selected.prediction_ia:,.0f}')
    cols[2].metric('Référence simple',f'{selected.prediction_reference:,.0f}')
    p=float(selected.proba_baisse)
    cols[3].metric('Score de baisse au mois suivant',f'{p:.1%}' if np.isfinite(p) else ('Premier mois uniquement' if mode!='Validation historique' and selected.Horizon>1 else 'Non calculable'))
    variation=selected.prediction_ia/selected.UN_SOG-1 if selected.UN_SOG>0 else None
    st.write('**Variation prévue du volume :** '+(f'{variation:+.1%}' if variation is not None else 'non calculable avec un volume initial nul'))
    signal_kind='Signal du classificateur' if pd.notna(selected.proba_baisse) else 'Baisse projetée par rapport au dernier mois connu'
    st.write('**'+signal_kind+' :** '+('à examiner' if selected.alerte else 'pas de signal, sans garantie d’absence de baisse'))
    st.caption(f'Classificateur de baisse future distinct de la régression de volume. Seuil de score retenu sur validation : {report["risk_threshold"]:.0%}. Probabilité non calibrée : elle ne garantit pas une fréquence réelle correspondante.')
    if pd.notna(selected.proba_baisse) and variation is not None and (variation<=-.2)!=bool(selected.alerte_classification):st.info('Le signal de classification et la baisse estimée par la régression divergent. Les examiner ensemble avant toute action.')
    h=allowed[(allowed.IdCodeEAN==pid)&(allowed.Date<=origin)].sort_values('Date')
    if mode=='Projection hors échantillon disponible':
        product_projection=projections[projections.IdCodeEAN==pid].sort_values('Date_cible')
    else:
        product_projection=portfolio[portfolio.IdCodeEAN==pid].copy()
    st.subheader('Historique et projection du produit')
    history_months=st.slider('Nombre de mois d’historique affichés',6,36,18,key='history_months')
    visible=h[h.Date>=origin-pd.DateOffset(months=history_months-1)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=visible.Date,y=visible.UN_SOG,name='Ventes observées',mode='lines+markers',line=dict(color='#1B2A4A',width=3)))
    future_x=[origin]+product_projection.Date_cible.tolist()
    fig.add_trace(go.Scatter(x=future_x,y=[selected.UN_SOG]+product_projection.prediction_ia.tolist(),name='Projection IA',mode='lines+markers',line=dict(color='#D9822B',dash='dash',width=4),marker_size=9))
    fig.add_trace(go.Scatter(x=product_projection.Date_cible,y=product_projection.prediction_saisonniere,name='Référence saisonnière',mode='lines+markers',line=dict(color='#2E8B7A',dash='dot',width=3)))
    fig.add_trace(go.Scatter(x=product_projection.Date_cible,y=product_projection.prediction_reference,name='Référence : dernier mois connu',mode='lines',line=dict(color='#8692A8',dash='dot')))
    fig.add_vline(x=origin.timestamp()*1000,line_dash='dot',line_color='#6B7280')
    fig.update_layout(xaxis_title='Mois',yaxis_title='Nombre de ventes',height=650,font_size=16,legend=dict(orientation='h',y=-.18),margin=dict(l=50,r=30,t=30,b=100),hovermode='x unified')
    fig.update_xaxes(dtick='M1',tickformat='%m/%Y',tickangle=-45)
    st.plotly_chart(fig,width='stretch',key='forecast_chart')
    st.caption('À gauche de la séparation : données observées. À droite : estimations ; aucune vente future réelle n’est ajoutée à la projection.')
    table=product_projection[['Date_cible','prediction_ia','prediction_saisonniere','prediction_reference']].copy()
    table['Date_cible']=table.Date_cible.dt.strftime('%Y-%m')
    table=table.rename(columns={'Date_cible':'Mois','prediction_ia':'Prévision IA','prediction_saisonniere':'Référence saisonnière','prediction_reference':'Dernier mois connu'})
    st.dataframe(table.round(0),hide_index=True,width='stretch')
    st.download_button('Télécharger les prévisions du produit (CSV)',table.to_csv(index=False,sep=';').encode('utf-8-sig'),file_name='previsions_produit.csv',mime='text/csv')
    st.subheader('Trois analyses du portefeuille filtré')
    st.caption(f'Laboratoire : {lab} · marque : {brand} · segment : {segment} · sous-segment : {subsegment}. Les graphiques portent sur les {len(portfolio)} présentations de cette sélection, pas sur tout le marché.')
    scope=projections[projections.IdCodeEAN.isin(portfolio.IdCodeEAN)] if mode=='Projection hors échantillon disponible' else portfolio
    dimension=st.selectbox('Répartition des ventes prévues selon',['Marque','Segment','SousSegment'],format_func=lambda v:'Sous-segment' if v=='SousSegment' else v,key='chart_dimension')
    totals=scope.groupby(dimension,as_index=False).prediction_ia.sum()
    bar=px.bar(totals,x=dimension,y='prediction_ia',color_discrete_sequence=['#D9822B'],labels={'prediction_ia':'Ventes prévues cumulées',dimension:'Sous-segment' if dimension=='SousSegment' else dimension},title='Répartition des ventes prévues sur la période affichée')
    bar.update_layout(height=460,font_size=15)
    st.plotly_chart(bar,width='stretch',key='forecast_distribution')
    st.caption('Les barres additionnent les ventes prévues de la période : six mois en projection, un mois en validation historique. Elles ne représentent pas des parts de marché nationales.')
    hist=px.histogram(portfolio,x='prediction_ia',nbins=min(20,max(1,len(portfolio))),color_discrete_sequence=['#2E8B7A'],labels={'prediction_ia':'Nombre de ventes prévues'},title='Distribution des ventes prévues par présentation — '+target.strftime('%Y-%m'))
    hist.update_layout(height=450,font_size=15,yaxis_title='Nombre de présentations')
    st.plotly_chart(hist,width='stretch',key='forecast_histogram')
    scatter=px.scatter(portfolio,x='UN_SOG',y='prediction_ia',color='Segment',hover_name='NomPresentation',labels={'UN_SOG':'Ventes du dernier mois connu','prediction_ia':'Ventes prévues'},title='Ventes connues et prévues par présentation — '+target.strftime('%Y-%m'))
    maximum=max(float(portfolio.UN_SOG.max()),float(portfolio.prediction_ia.max()),1)
    scatter.add_trace(go.Scatter(x=[0,maximum],y=[0,maximum],mode='lines',name='Ventes inchangées',line=dict(color='#8692A8',dash='dot')))
    scatter.update_layout(height=500,font_size=15)
    st.plotly_chart(scatter,width='stretch',key='forecast_scatter')
    st.caption('Au-dessus de la diagonale : hausse projetée ; en dessous : baisse projetée. Ces points représentent des estimations, pas une relation causale. Avec une seule présentation, élargir les filtres pour comparer plusieurs produits.')
    if mode=='Validation historique' and st.checkbox('Afficher le résultat réellement observé après la date de prévision',key='reveal'):
        st.metric('Ventes réellement observées',f'{selected.ventes_futures:,.0f}')
        st.write(f'Erreur absolue IA : {abs(selected.prediction_ia-selected.ventes_futures):,.1f} ; référence : {abs(selected.prediction_reference-selected.ventes_futures):,.1f}.')
        st.caption('La vérité future est affichée uniquement pour examiner le test historique, et n’est pas une variable du modèle.')

elif page=='Alertes portefeuille':
    st.subheader('Références à examiner pour '+target.strftime('%Y-%m'))
    alerts=portfolio[portfolio.UN_SOG>0].sort_values('prediction_ia',ascending=True).copy()
    alerts['Variation prévue (%)']=100*(alerts.prediction_ia/alerts.UN_SOG-1)
    alerts=alerts.sort_values('proba_baisse',ascending=False)
    signal_label='du classificateur à un mois' if portfolio.proba_baisse.notna().any() else 'de baisse projetée ≥ 20 % face au dernier mois connu'
    st.write(f'{int(alerts.alerte.sum())} signal(s) {signal_label} parmi {len(alerts)} références éligibles. Le seuil métier et le budget d’alertes restent à valider avec le client.')
    display=alerts[['NomPresentation','Marque','UN_SOG','prediction_ia','proba_baisse','alerte','Variation prévue (%)']].rename(columns={'UN_SOG':'Volume connu','prediction_ia':'Prévision IA','proba_baisse':'Score de baisse à un mois','alerte':'Signal'})
    st.dataframe(display,hide_index=True,width='stretch')
    st.download_button('Exporter la revue portefeuille (CSV)',display.to_csv(index=False,sep=';').encode('utf-8-sig'),file_name='revue_portefeuille.csv',mime='text/csv')
    st.write('**Revue proposée :** vérifier la saisonnalité et la qualité des données ; confronter le signal aux stocks, lancements et actions commerciales ; retenir une action seulement après discussion métier.')
    st.caption('Les valeurs nulles à la date de référence rendent le risque relatif de baisse non interprétable ; ces références sont exclues des alertes.')

elif page=='Évaluation IA':
    st.subheader('Test chronologique exploratoire')
    st.write(f'{report["train"]:,} paires d’apprentissage ({report["train_pct"]:.1%}) ; {report["test"]:,} paires de test. Validation : mai–octobre 2025 ; test : novembre 2025–juin 2026. Les modèles et les seuils sont choisis sur la validation, puis figés pour le test.')
    regtable=pd.DataFrame(report['reg_test']).T
    st.dataframe(regtable[['MAE','RMSE','MSE','WAPE','R2_unites']].round(3),width='stretch')
    rname=report['reg_selected'];gain=1-report['reg_test'][rname]['MAE']/report['reg_test']['Persistance']['MAE']
    st.write(f'Gain de MAE face à la référence simple sur ce test : {gain:+.1%}. Ce score ne démontre pas un gain commercial ni une robustesse sur d’autres marchés.')
    st.warning('La référence saisonnière obtient une meilleure MAE et une meilleure RMSE que la forêt sur ce test. La supériorité globale de l’IA n’est pas démontrée. Le classificateur signale 68,6 % des références éligibles sur le test : une charge élevée à revoir avec le métier.')
    risk=pd.DataFrame({k:{m:v for m,v in item.items() if m in ['Accuracy','Balanced_accuracy','Precision','Recall','F2','AP','Alerts']} for k,item in report['risk_test'].items()}).T
    st.dataframe(risk.round(3),width='stretch')
    cm=report['risk_test'][report['risk_selected']]['Confusion'];st.dataframe(pd.DataFrame(cm,index=['Réel : pas de baisse ≥ 20 %','Réel : baisse ≥ 20 %'],columns=['Pas de signal','Signal']),width='stretch')
    st.write('Un faux positif déclenche une revue inutile. Un faux négatif laisse passer une baisse. Le rappel et la précision doivent être examinés ensemble, avec la capacité de traitement des utilisateurs.')
    st.caption('Le protocole a été affiné pendant la reprise ; ce test est exploratoire et ne remplace pas un nouveau test prospectif. Scores globaux sur le marché, pas scores spécifiques garantis pour le laboratoire choisi. Les probabilités ne sont pas calibrées. Les prix/facings sont exclus, mais les traitements initiaux de la base peuvent encore affecter la validité temporelle.')
    if (ROOT/'evaluation_6mois.json').exists():
        st.subheader('Évaluation de la projection à six mois')
        six_report=json.loads((ROOT/'evaluation_6mois.json').read_text())
        st.write('Projection faite depuis octobre 2025 pour novembre 2025 à avril 2026, sans actualisation par les ventes réelles entre les horizons. Ces résultats sont distincts du test à un mois présenté ci-dessus.')
        rows=[]
        for row in six_report['resultats']:
            rows.append({'Horizon':row['Horizon'],'Mois cible':row['Mois'],'Présentations évaluées':row['Observations'],'MAE IA':row['prediction_ia']['MAE'],'MAE dernier mois connu':row['prediction_reference']['MAE'],'MAE saisonnière':row['prediction_saisonniere']['MAE']})
        st.dataframe(pd.DataFrame(rows).round(1),hide_index=True,width='stretch')
        st.warning('Sur cet essai à origine fixe, la référence saisonnière a une erreur moyenne plus faible que l’IA à chacun des six horizons. La projection IA à six mois reste expérimentale ; les scores à un mois ne démontrent pas sa fiabilité à six mois.')
        st.caption(six_report['statut'])
    with st.expander('Comparaison sur validation et paramètres'):
        st.json({'reg_validation':report['reg_validation'],'risk_validation':report['risk_validation'],'policy':report['threshold_policy']})

elif page=='Dossier de décision':
    st.subheader('Préparer une discussion métier')
    st.write('Les notes sont conservées uniquement pendant la session. Le PDF est généré en mémoire ; son téléchargement crée une copie sur ton ordinateur.')
    contexte=st.text_area('Vérifications métier et contexte connu',placeholder='Saisonnalité, stock, lancement, actions récentes…',key='note_'+mode+str(origin)+str(pid))
    statut=st.selectbox('Suite proposée',['À examiner','Demander une vérification des données','Préparer un échange métier','Reporter faute de données suffisantes'])
    lines=[f'Laboratoire : {lab} — marque : {selected.Marque}',f'Présentation : {selected.NomPresentation}',f'Données connues jusqu’à {origin:%Y-%m} ; cible {target:%Y-%m}',f'Usage : {mode}',f'Volume connu : {selected.UN_SOG:,.1f} ; prévision IA : {selected.prediction_ia:,.1f} ; référence simple : {selected.prediction_reference:,.1f}',f'Score exploratoire de baisse ≥ 20 % : {selected.proba_baisse:.1%}' if pd.notna(selected.proba_baisse) else ('Score de baisse réservé au premier mois de projection.' if mode=='Projection hors échantillon disponible' and selected.Horizon>1 else 'Risque relatif non calculable avec le volume initial disponible.'),(f'Signal du classificateur : {bool(selected.alerte)} ; seuil de score {report["risk_threshold"]:.0%}' if pd.notna(selected.proba_baisse) else f'Signal de baisse projetée face au dernier mois connu : {bool(selected.alerte)} ; seuil de baisse 20 %'),f'Suite proposée : {statut}',f'Contexte et vérifications : {contexte or "non renseignés"}','Limites : prototype sans validation métier ; score de baisse disponible au premier mois uniquement ; projection récursive à hypothèses constantes ; aucune recommandation automatique de linéaire ni effet causal démontré.']
    if mode=='Projection hors échantillon disponible':
        lines.append(f'Projection de {origin+pd.DateOffset(months=1):%Y-%m} à {origin+pd.DateOffset(months=6):%Y-%m} ; ventes récursives, distribution, VMM et stock constants à la date d’origine.')
        for row in projections[projections.IdCodeEAN==pid].sort_values('Date_cible').itertuples():
            lines.append(f'{row.Date_cible:%Y-%m} : prévision IA {row.prediction_ia:,.0f} ; référence saisonnière {row.prediction_saisonniere:,.0f}')
    st.download_button('Télécharger le dossier PDF',pdf_report('Anticipation commerciale — Dossier de décision',lines),file_name='dossier_decision.pdf',mime='application/pdf')

elif page=='Administration':
    st.subheader('Contrôles administrateur')
    st.write('Accès réservé au rôle admin par le code serveur. Les comptes clients sont limités à leur laboratoire pour les observations, prédictions, graphiques et exports.')
    st.json({'source_rows':report['rows_source'],'products':report['products'],'train':report['train'],'test':report['test'],'reg_selected':report['reg_selected'],'risk_selected':report['risk_selected']})
    st.write('Tables SQL disponibles :')
    with sqlite3.connect(f'file:{ROOT / "nasal_inhalation.db"}?mode=ro',uri=True) as c:
        names=c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        st.dataframe(pd.DataFrame(names,columns=['Table']),hide_index=True)
    st.caption('Protection limitée au prototype : PBKDF2 et contrôle des rôles. Le chiffrement disque, TLS, limitation des tentatives, journalisation sécurisée et OIDC nécessitent une intégration de production. Le fichier local de données n’est pas chiffré par cette application.')
