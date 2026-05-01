"""
Este código implementa uma aplicação web interativa de análise de score de crédito utilizando a biblioteca Streamlit. A interface permite ao usuário inserir informações financeiras e pessoais do cliente, como idade, renda mensal, nível de endividamento e histórico de atrasos.

Os dados são coletados por meio de controles interativos (sliders e campos numéricos) na barra lateral. Após o preenchimento, o usuário pode clicar em um botão para enviar essas informações a uma API local de previsão.

A aplicação então realiza uma requisição HTTP para o endpoint configurado, que retorna a classificação de risco (inadimplente ou não inadimplente) e a probabilidade associada. Com base nesse resultado, o sistema exibe de forma clara se o cliente apresenta alto risco ou baixo risco de inadimplência, junto com a probabilidade estimada.

Além disso, a interface apresenta métricas resumidas e organiza os resultados de maneira visual, facilitando a interpretação e auxiliando na tomada de decisão.
"""

import streamlit as st
import requests
from dotenv import load_dotenv
import os

st.set_page_config(page_title="credit scoring", page_icon="💳")
col1, col2 = st.columns(2, border=True)
col1.page_link("pages/app.py", label="Score de Crédito (1º Painel)", icon="💳")
col2.page_link("pages/painel.py", label="Dashboard de Risco (2º Painel)", icon="📊")

load_dotenv()

API_URL=os.getenv("API_URL")

st.title("💰 Score de Crédito 💰", text_alignment="center")

st.sidebar.markdown("# 📊 Dados do Cliente")

st.sidebar.markdown("## Utilize as opções abaixo⬇️")

st.sidebar.divider()

taxa = st.sidebar.slider(
    label="Taxa De Utilizacao De Linhas Nao Garantidas", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5
    )

idade = st.sidebar.slider(
    label="Idade",
    min_value=18,
    max_value=109,
    value=30
    )

num_30_59 = st.sidebar.number_input(
    label="Numero De Vezes 30-59 Dias Atraso",
    min_value=0,
    max_value=20,
    value=0
)

taxa_de_endividamento = st.sidebar.slider(
    label="Taxa De Endividamento",
    min_value=0.0,
    max_value=12.0,
    value=0.5
)

renda_mensal = st.sidebar.number_input(
    label="Renda Mensal",
    min_value=0,
    value=2000
)

numero_de_credito = st.sidebar.number_input(
    label="Numero De Linhas De Credito E Emprestimos Abertos",
    min_value=0,
    max_value=58,
    value=4
)

numero_de_vezes_90_dias_atraso = st.sidebar.number_input(
    label="Numero De Vezes 90 Dias Atraso",
    min_value=0,
    max_value=20,
    value=0
)

num_emprestimos_ou_linhas_imobiliarias = st.sidebar.slider(
    label="Numero De Emprestimos Ou Linhas Imobiliarias",
    min_value=0,
    max_value=54,
    value=0
)

num_vezes_60_89_dias_atraso = st.sidebar.number_input(
    label="Numero De Vezes 60-89 Dias Atraso",
    min_value=0,
    max_value=20,
    value=0
)

num_de_dependentes = st.sidebar.number_input(
    label="Numero de Dependentes",
    min_value=0,
    max_value=20,
    value=0
)

col1, col2 = st.columns(2, border=True)

col1.metric("Renda:", f"R${renda_mensal:,.0f}".replace(",","."))
col2.metric("Taxa de Endividamento:", f"{taxa_de_endividamento:.2f}")

st.divider()

st.markdown("# ANÁLISE PREDITIVA", text_alignment="center")
predict_btn = st.button("⏳ Rodar Previsão 👈clique aqui", use_container_width=True)

if predict_btn:
    payload = {
       'TaxaDeUtilizacaoDeLinhasNaoGarantidas': taxa,
       'Idade': idade, 
       'NumeroDeVezes30_59DiasAtrasoNaoPior': num_30_59, 
       'TaxaDeEndividamento': taxa_de_endividamento,
       'RendaMensal': renda_mensal, 
       'NumeroDeLinhasDeCreditoEEmprestimosAbertos': numero_de_credito,
       'NumeroDeVezes90DiasAtraso': numero_de_vezes_90_dias_atraso, 
       'NumeroDeEmprestimosOuLinhasImobiliarias': num_emprestimos_ou_linhas_imobiliarias,
       'NumeroDeVezes60_89DiasAtrasoNaoPior': num_vezes_60_89_dias_atraso, 
       'NumeroDeDependentes': num_de_dependentes
    }

    with st.spinner("Consultando o modelo."):
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                pred = result["predict"][0]
                probabilidade = result["probabilidade"][0][pred]

                st.subheader("📈 Resultado da Análise")

                if pred == 1:
                    risco = "🔴 Alto Risco"
                    texto = "### Chance do devedor não quitar a dívida"
                    cor = "red"
                    pred = "inadimplente"
                else:
                    risco = "🟢 Baixo Risco"
                    texto = "### Chance do devedor quitar a dívida"
                    cor = "green"
                    pred = "não inadimplente"

                st.markdown(f"### **:{cor}[{risco}]**")

                st.markdown(texto)

                st.write(f"Resultado: **{pred}** com probabilidade **{probabilidade:.2%}**")

        except Exception as erro:
            st.error(f"Erro ao conectar com a API: {erro}")













