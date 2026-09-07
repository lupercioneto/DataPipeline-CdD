import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from logs.log import Logger, LogType


class ClassificationAnalyzer:
    """
    Pipeline de classificação binária (SKLearn) comparando Regressão
    Logística e K-Vizinhos Mais Próximos (KNN), com pré-processamento
    (padronização) e otimização de hiperparâmetros via GridSearchCV.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
        target_source_column: str,
        target_column: str = "is_fast_vax",
        test_size: float = 0.3,
        random_state: int = 42,
    ):
        self.feature_columns = feature_columns
        self.target_source_column = target_source_column
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

        self.dataframe = dataframe.copy()

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.grids: dict[str, GridSearchCV] = {}
        self.results: dict[str, dict] = {}

    def prepare_data(self):
        """Define a tarefa de classificação binária (via corte na mediana
        da coluna-fonte) e separa treino/teste."""

        Logger.alert(
            f"Definindo alvo binário '{self.target_column}' a partir da mediana de "
            f"'{self.target_source_column}'...",
            LogType.INFO,
        )

        median_value = self.dataframe[self.target_source_column].median()
        self.dataframe[self.target_column] = (
            self.dataframe[self.target_source_column] > median_value
        ).astype(int)

        X = self.dataframe[self.feature_columns].fillna(0)
        y = self.dataframe[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        Logger.alert(
            f"Dados preparados: treino n={len(self.X_train)}, teste n={len(self.X_test)}",
            LogType.SUCCESS,
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def run_grid_search(self):
        """Compara Regressão Logística e KNN, cada um em um pipeline com
        StandardScaler, otimizando hiperparâmetros via GridSearchCV
        (Validação Cruzada, cv=5)."""

        if self.X_train is None:
            raise RuntimeError("Execute prepare_data() antes do GridSearchCV.")

        Logger.alert(
            "Executando GridSearchCV (cv=5) para Regressão Logística e KNN...",
            LogType.INFO,
        )

        pipe_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
        pipe_knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])

        param_grid_lr = {"lr__C": [0.1, 1.0, 10.0]}
        param_grid_knn = {"knn__n_neighbors": [3, 5, 7]}

        self.grids = {
            "Regressão Logística": GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1"),
            "KNN": GridSearchCV(pipe_knn, param_grid_knn, cv=5, scoring="f1"),
        }

        for name, grid in self.grids.items():
            grid.fit(self.X_train, self.y_train)
            Logger.alert(
                f"{name}: melhores hiperparâmetros = {grid.best_params_}",
                LogType.SUCCESS,
            )

        return self.grids

    def evaluate(self):
        """Reporta a Matriz de Confusão, Acurácia, Precisão, Sensibilidade
        (Recall) e F1-Score de cada modelo no conjunto de teste."""

        if not self.grids:
            raise RuntimeError("Execute run_grid_search() antes de avaliar os modelos.")

        for name, grid in self.grids.items():
            y_pred = grid.predict(self.X_test)

            metrics = {
                "best_params": grid.best_params_,
                "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1_score": f1_score(self.y_test, y_pred),
            }
            self.results[name] = metrics

            Logger.alert(
                f"[{name}] Matriz de Confusão: {metrics['confusion_matrix']} | "
                f"Acurácia={metrics['accuracy']:.4f} | Precisão={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | F1-Score={metrics['f1_score']:.4f}",
                LogType.SUCCESS,
            )

        return self.results

    def run_full_analysis(self):

        self.prepare_data()
        self.run_grid_search()
        results = self.evaluate()

        return {
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "results": results,
        }


if __name__ == "__main__":
    # Execução standalone: espera ser rodado a partir da pasta `src`
    # (mesma convenção do main.py: `cd src && python models/machine_learning.py`)
    from extract.extractor import DataExtractor

    extractor = DataExtractor("../dados_limpos_final.csv")
    df = extractor.extract_csv()

    if df is None:
        Logger.alert(
            "Não foi possível carregar 'dados_limpos_final.csv'. Execute main.py primeiro.",
            LogType.WARNING,
        )
    else:
        analyzer = ClassificationAnalyzer(
            dataframe=df,
            feature_columns=[
                "total_vaccinations_per_hundred",
                "people_vaccinated_per_hundred",
                "people_fully_vaccinated_per_hundred",
            ],
            target_source_column="daily_vaccinations_per_million",
            target_column="is_fast_vax",
        )

        results = analyzer.run_full_analysis()

        Logger.alert(f"Resumo Classificação: {results}", LogType.INFO)
