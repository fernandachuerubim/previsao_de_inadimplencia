from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel
import dagshub
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD']

mlflow.set_tracking_uri("https://dagshub.com/fernandachuerubim/previsao_de_inadimplencia.mlflow")

model_name = "credit_scoring_model"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri=model_uri) # pegando o caminho que ta no mlflow

class DadosEntrada(BaseModel):
    """
    Essa classes traz todas as colunas da base de dados.

    Attributes:
        TaxaDeUtilizacaoDeLinhasNaoGarantidas (float):  Proporção do uso de crédito não garantido (ex: cartões de crédito) em relação ao limite total disponível.

        Idade (int): Idade do indivíduo em anos.

        NumeroDeVezes30_59DiasAtrasoNaoPior (int): Quantidade de vezes em que houve atraso entre 30 e 59 dias em pagamentos, sem atrasos mais graves associados.
        
        TaxaDeEndividamento (float): Relação entre o total de dívidas e a renda mensal do indivíduo.

        RendaMensal (float): Renda mensal do indivíduo.

        NumeroDeLinhasDeCreditoEEmprestimosAbertos (int): Número total de contas de crédito e empréstimos atualmente abertas.

        NumeroDeVezes90DiasAtraso (int): Número de vezes em que o indivíduo teve atraso superior a 90 dias.

        NumeroDeEmprestimosOuLinhasImobiliarias (int): Quantidade de empréstimos ou linhas de crédito relacionadas a imóveis.

        NumeroDeVezes60_89DiasAtrasoNaoPior (int): Número de vezes em que houve atraso entre 60 e 89 dias, sem atrasos mais graves associados.

        NumeroDeDependentes (int): Número de pessoas financeiramente dependentes do indivíduo.
    """

    TaxaDeUtilizacaoDeLinhasNaoGarantidas : float
    Idade : int
    NumeroDeVezes30_59DiasAtrasoNaoPior: int
    TaxaDeEndividamento: float
    RendaMensal: float
    NumeroDeLinhasDeCreditoEEmprestimosAbertos: int
    NumeroDeVezes90DiasAtraso: int
    NumeroDeEmprestimosOuLinhasImobiliarias: int
    NumeroDeVezes60_89DiasAtrasoNaoPior: int
    NumeroDeDependentes: int

@app.get("/")
def home():
    """
    Mostra o status do API rodando.
    """
    return{"Status": "API Rodando"}

@app.post("/predict")
def predict(dados: DadosEntrada):
    """
    Mostra a última versão do melhor do modelo no MLflow.

    Returns:
        data (dict): dicionário com a predição e a probabilidade.
    """

    df = pd.DataFrame([dados.model_dump()])

    yhat = model.predict(df)
    proba = model.predict_proba(df)

    data = {"predict": yhat.tolist(),
            "probabilidade": proba.tolist()
            } # está transformando um resultado numa lista
    return data






