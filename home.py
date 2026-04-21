"""
O código configura uma aplicação Streamlit com múltiplas páginas utilizando navegação personalizada.
"""

import streamlit as st

score = st.Page("pages/app.py", title="Score de Crédito (1º Painel)", icon="💳")
modelo = st.Page("pages/painel.py", title="Dashboard (2º Painel)", icon="📊")

pg = st.navigation([score, modelo])
pg.run()