# Descrição 
Projeto de Análise Preditiva de Crédito com Treinamento de Modelo, API de previsão e dashboard.

Este repositório contém um pipeline para treinar modelos de score de crédito usando MLflow, além de uma API para fazer as predições e um painel streamlit para interagir com os dados.

# Estrutura do Projeto
- `main.py` - Treina os dados, usa o optuna para otimizar os parâmetros do modelo para que tenhamos os melhores resultados e registra o melhor o modelo no MLflow.

- `api.py` - API utilizando o FastAPI e expõe a rota `/predict`. 

- `home.py` - Cria as páginas no streamlit.

- `pages/painel.py` - Utiliza métricas e cria dashaboards.

- `pages/app.py` - Utiliza os dados para fazer a predição do modelo.

- `pyproject.toml` - Utiliza várias dependências para o desenvolvimento do projeto.

# Ferramentas
- Python(pandas, sklearn, plotly)
- Streamlit
- VS Code
- FastAPI
- MLflow
- Optuna

# Desenvolvimento do Projeto
## 1ª Etapa
- Carrega os dados da pasta data;
- Divide os dados em treino e validação;
- Cria o pipeline para o pré-processamento dos dados;
- Utiliza o optuna para encontrar os melhores parâmetros para a escolha do melhor modelo.
- Faz o registro do melhor modelo no framework MLflow.

## 2ª Etapa
- Implantação da API(Interface de Programação de Aplicativos) para disponibilizar a predição do melhor modelo.
- Cria a rota predict para mostrar os parâmetros da predição do modelo.

## 3ª Etapa
- Criação dos paineis na página do streamlit;
- 1º painel: informações de métricas do score de crédito e predição do modelo;
- 2º painel: informações de métricas do dashboard de score de crédito, gráfico de barras dos clientes inadimplentes, gráfico de barras por faixa etária, entre outras informações analisadas.

# Resultados Obtidos
- Dashboard visualizado no streamlit;
- API.
