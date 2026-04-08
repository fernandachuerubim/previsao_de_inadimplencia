# %%
import pandas as pd
import os
from sklearn.model_selection import train_test_split # divide os dados em treino e teste
from sklearn.impute import SimpleImputer #imputa dados de colunas que apresentam dados faltantes
from sklearn.preprocessing import StandardScaler # normaliza os dados
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# %%
path_home = os.getcwd()
train_path = os.path.join(path_home, "data", "train.csv")
df = pd.read_csv(train_path)
# df = pd.read_csv("data\\train.csv")
# df = pd.read_csv(r"data\\train.csv")

# %%
columns = ['target', 'TaxaDeUtilizacaoDeLinhasNaoGarantidas',
       'Idade', 'NumeroDeVezes30-59DiasAtrasoNaoPior', 'TaxaDeEndividamento',
       'RendaMensal', 'NumeroDeLinhasDeCreditoEEmprestimosAbertos',
       'NumeroDeVezes90DiasAtraso', 'NumeroDeEmprestimosOuLinhasImobiliarias',
       'NumeroDeVezes60-89DiasAtrasoNaoPior', 'NumeroDeDependentes']

df = df[columns]
# %%
X = df.drop("target", axis=1)
y = df["target"]

# %%
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.3
)

# %%
column_median = ["RendaMensal", "NumeroDeDependentes"]
remover = ["target", "RendaMensal", "NumeroDeDependentes"]

column_scaler = [col for col in columns if col not in remover]

# %%
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

scaler = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", 
         numeric_transformer, 
        column_median),
        (
            "scaler", # nome da coluna
            scaler, # é o pipeline
            column_scaler, # é a lista de coluna
        ),
    ],
    remainder="passthrough" # significa que será preservada as colunas que não vão sofrer transformação
)

# %%
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced"))
])

# %%
pipeline.fit(X_train, y_train)

# %%
pred = pipeline.predict_proba(X_valid)[:, 1]

# %%
metric = roc_auc_score(y_valid, pred)

# %%
