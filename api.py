from fastapi import FastAPI
import mlflow
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

class DadosEntrada(BaseModel):
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
    return{"Status": "API Rodando"}

@app.post("/predict")
def predict(dados: DadosEntrada):

    model_name = "credit_scoring_model"
    model_version = "latest"

    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.sklearn.load_model(model_uri=model_uri) # pegando o caminho que ta no mlflow

    df = pd.DataFrame([dados.model_dump()])

    yhat = model.predict(df)
    proba = model.predict_proba(df)

    return {"predict": yhat.tolist(),
            "probabilidade": proba.tolist()
            } # está transformando um resultado numa lista






