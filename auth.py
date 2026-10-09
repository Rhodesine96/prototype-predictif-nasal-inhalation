"""Contrôle d’accès local de démonstration, mots de passe PBKDF2, rôles serveur."""
from pathlib import Path
import hashlib,hmac,json
import streamlit as st

def authenticate(root):
    try:
        accounts=json.loads(st.secrets['accounts_json'])
    except Exception:
        st.error('Les comptes doivent être configurés dans les secrets Streamlit.')
        st.stop()
    if 'account' not in st.session_state:
        st.subheader('Connexion au prototype')
        with st.form('login'):
            name=st.text_input('Identifiant')
            password=st.text_input('Mot de passe',type='password')
            submit=st.form_submit_button('Se connecter')
        if submit:
            item=accounts.get(name)
            digest=hashlib.pbkdf2_hmac('sha256',password.encode(),bytes.fromhex(item['salt']),300000).hex() if item else ''
            if item and hmac.compare_digest(digest,item['hash']):
                st.session_state['account']={'id':name,'role':item['role'],'laboratoire':item.get('laboratoire')}
                st.rerun()
            else:st.error('Identifiant ou mot de passe incorrect.')
            st.stop()
    account=st.session_state['account']
    st.sidebar.caption('Compte : '+account['id']+' · rôle : '+account['role'])
    if st.sidebar.button('Se déconnecter'):
        st.session_state.clear();st.rerun()
    return account
