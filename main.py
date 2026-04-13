
import pandas as pd
import os
import optuna
import mlflow
from sklearn.model_selection import train_test_split, cross_val_score  # divide os dados em treino e teste
from sklearn.impute import SimpleImputer #imputa dados de colunas que apresentam dados faltantes
from sklearn.preprocessing import StandardScaler # normaliza os dados
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from mlflow.tracking import MlflowClient


class DataLoader():
    def load(self):
        # Carregando os dados
        path_home = os.getcwd()
        train_path = os.path.join(path_home, "data", "train.csv")
        df = pd.read_csv(train_path)

        columns = ['target', 'TaxaDeUtilizacaoDeLinhasNaoGarantidas',
       'Idade', 'NumeroDeVezes30-59DiasAtrasoNaoPior', 'TaxaDeEndividamento',
       'RendaMensal', 'NumeroDeLinhasDeCreditoEEmprestimosAbertos',
       'NumeroDeVezes90DiasAtraso', 'NumeroDeEmprestimosOuLinhasImobiliarias',
       'NumeroDeVezes60-89DiasAtrasoNaoPior', 'NumeroDeDependentes']

        df = df[columns]
        # Separando os dados em X e y
        X = df.drop("target", axis=1)
        y = df["target"]

        # Fazendo o split dos dados
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            stratify=y,
            test_size=0.3
        )
        return X_train, X_valid, y_train, y_valid

class PreprocessorBuilder:
    def build(self):
        columns = ['target', 'TaxaDeUtilizacaoDeLinhasNaoGarantidas',
       'Idade', 'NumeroDeVezes30-59DiasAtrasoNaoPior', 'TaxaDeEndividamento',
       'RendaMensal', 'NumeroDeLinhasDeCreditoEEmprestimosAbertos',
       'NumeroDeVezes90DiasAtraso', 'NumeroDeEmprestimosOuLinhasImobiliarias',
       'NumeroDeVezes60-89DiasAtrasoNaoPior', 'NumeroDeDependentes']

        # Criando as colunas para o pipeline
        column_median = ["RendaMensal", "NumeroDeDependentes"]
        remover = ["target", "RendaMensal", "NumeroDeDependentes"]

        column_scaler = [col for col in columns if col not in remover]

    # Criando o pipeline
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        scaler = Pipeline(steps=[("scaler", StandardScaler())]) # a normalização é realizada em todas as colunas

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

        return preprocessor

class ModelTrainer:
    def __init__(self, X_train, y_train, X_valid, y_valid, preprocessor):
        self.X_train=X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.preprocessor = preprocessor

        self.client = MlflowClient()
        self.MODEL_NAME = "credit_scoring_model"
        # Configurações
        mlflow.set_experiment("credit_scoring_optuna") # nome do projeto ou nome do experimento

# Criando a função objective para ser otimizada pelo optuna
    def objective(self, trial: optuna.trial.Trial):
        with mlflow.start_run(nested=True):
            model_name = trial.suggest_categorical("model", [
                "RandomForest", 
                # "GradientBoostingClassifier", 
                "LogisticRegression"])
            if model_name == "RandomForest": 
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("rf_n_estimators", 50, 500), 
                    max_depth=trial.suggest_int("rf_max_depth", 3,12), 
                    min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
                    class_weight="balanced"
                )
            
            elif model_name == "GradientBoostingClassifier":
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("gb_n_estimators", 50, 500), 
                    max_depth=trial.suggest_int("gb_max_depth", 3,12),
                    learning_rate=trial.suggest_float("gb_learning_rate", 0.01, 0.3)
                )

            else:
                model = LogisticRegression(
                    C=trial.suggest_float("lr_c", 0.01, 10),
                    max_iter=trial.suggest_int("lr_max_iter", 100, 2000),
                    class_weight="balanced",
                )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", self.preprocessor),
                    ("model", model)
            ])

            result = cross_val_score(pipeline, self.X_train, self.y_train, cv=3, scoring="roc_auc").mean()   

            mlflow.log_params(trial.params) #loga os parametros escolhidos o trial é do optuna
            mlflow.log_metric("roc_auc", result)

            pipeline.fit(self.X_train, self.y_train)

            model_info = mlflow.sklearn.log_model(pipeline, name=model_name)

            trial.set_user_attr("model_uri", model_info.model_uri) # cria um novo atributo no trial

            return result

    def optimaze(self, n_trial=6):
        # Inicilizando o mlflow e usando o optuna para otimizar
        with mlflow.start_run(run_name="optuna_optimization"):
            study = optuna.create_study(direction="maximize") # padrao optuna chama de study para maximizar
            study.optimize(self.objective, n_trials=n_trial)

        self.study = study
        return study
    
    # Cria a função para métrica ROC-AUC
    def evaluate_model(self, model_uri, X_valid, y_valid):
        pipeline = mlflow.sklearn.load_model(model_uri)
        pred = pipeline.predict_proba(X_valid)[:,1]

        return roc_auc_score(y_valid, pred)
    
    # Pega a versão do modelo com status de produção(production)
    def get_champion(self):
        versions = self.client.search_model_versions(f"name='{self.MODEL_NAME}'")
        for v in versions:
            if v.tags.get("status") == "production":
                return v
        return None

    def promoter_model(self):
        # Pegando o melhor trial do optuna
        best_trial = self.study.best_trial # pegando o melhor trial

        # pegando a uri do melhor modelo
        model_uri = best_trial.user_attrs["model_uri"]

        # Calcula a métrica ROC-AUC do modelo atual(desafiador)
        challenger_score = self.evaluate_model(
            model_uri=model_uri, 
            X_valid=self.X_valid, 
            y_valid=self.y_valid
            )

        # Pega o melhor modelo em produção
        champion = self.get_champion() # champion vai retornar a versão do mlflow

        # Verifica se existe um modelo em produção
        if champion is None:
            # Registra o modelo treinado
            result = mlflow.register_model(
                model_uri=model_uri, 
                name=self.MODEL_NAME
                )
            # Pega a versão do modelo
            new_version = result.version

            # Registra a versão treinada como produção
            self.client.set_model_version_tag(
                name=self.MODEL_NAME,
                version=new_version,
                key="status",
                value="production"
            )

            print("Primeiro modelo promovido para produção")

        else: # Caso exista um champion em produção
            # Cria a uri do champion
            champion_uri = f"models:/{self.MODEL_NAME}/{champion.version}"
            # Calcula a métrica ROC-AUC do modelo em produção
            champion_score = self.evaluate_model(
                model_uri=champion_uri,
                X_valid=self.X_valid,
                y_valid=self.y_valid
            )

            print(f"Champion ROC-AUC: {champion_score:.4f}")

            # Verifica qual modelo tem a melhor métrica (treinado ou produção)
            if challenger_score > champion_score + 0.05:
                print("Challenger é melhor! promovendo.")

                # Registra o modelo treinado
                result = mlflow.register_model(
                    model_uri=model_uri, 
                    name=self.MODEL_NAME
                    )
                # Pega a versão do modelo
                new_version = result.version

                # Registra a versão treinada como produção
                self.client.set_model_version_tag(
                    name=self.MODEL_NAME,
                    version=new_version,
                    key="status",
                    value="production"
                )
            
                # arquivando o champion no mlflow
                self.client.set_model_version_tag(
                    name=self.MODEL_NAME,
                    version=champion.version,
                    key="status",
                    value="archived"
                )
            # o modelo que está no mlflow é melhor que o treinado atualmente
            else:
                print("Champion continua sendo o melhor.")

if __name__ == "__main__":

    loader = DataLoader() #inicializando a classe
    X_train, X_valid, y_train, y_valid = loader.load() # carregando os dados

    preprocessor = PreprocessorBuilder().build() # carregando o preprocessor

    trainer = ModelTrainer( # criação do treino
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        preprocessor=preprocessor
        )

    study = trainer.optimaze(n_trial=6) # pegando os melhores parâmetros

    # Mostrando os melhores resultados da otimização
    print(f"Melhor ROC-AUC: {study.best_value}")
    print(f"Melhores parâmetros: {study.best_params}")

    trainer.promoter_model() # verifica qual modelo é o melhor se é o atual ou que está em produção










