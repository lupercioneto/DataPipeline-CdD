import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from logs.log import Logger, LogType


class RegressionAnalyzer:
    """
    Ajusta um modelo de Regressão Linear Múltipla e reporta a qualidade
    do ajuste (R² e RMSE), preservando a interpretação Ceteris Paribus
    dos coeficientes estimados (β0, β1, β2, ...).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        response_column: str,
        predictor_columns: list[str],
    ):
        self.response_column = response_column
        self.predictor_columns = predictor_columns

        clean_df = dataframe[[response_column, *predictor_columns]].dropna()

        self.X = clean_df[predictor_columns]
        self.y = clean_df[response_column]

        self.model = LinearRegression()
        self.y_pred = None
        self.r2 = None
        self.rmse = None

    def fit(self):
        """Ajusta a Regressão Linear Múltipla: Y = β0 + β1*X1 + β2*X2 + ..."""

        Logger.alert(
            f"Ajustando Regressão Linear Múltipla: {self.response_column} ~ "
            f"{' + '.join(self.predictor_columns)}",
            LogType.INFO,
        )

        self.model.fit(self.X, self.y)
        self.y_pred = self.model.predict(self.X)

        Logger.alert("Modelo de regressão ajustado com sucesso!", LogType.SUCCESS)

        return self.model

    def evaluate(self):
        """Calcula o Coeficiente de Determinação (R²) e a Raiz do Erro
        Quadrático Médio (RMSE) do ajuste."""

        if self.y_pred is None:
            raise RuntimeError("Execute fit() antes de avaliar o modelo.")

        self.r2 = r2_score(self.y, self.y_pred)
        self.rmse = float(np.sqrt(mean_squared_error(self.y, self.y_pred)))

        Logger.alert(f"R² = {self.r2:.4f} | RMSE = {self.rmse:.2f}", LogType.SUCCESS)

        return self.r2, self.rmse

    def report(self):
        """Reporta e interpreta os coeficientes estimados sob a ótica do
        princípio do Ceteris Paribus (tudo o mais constante)."""

        intercept = float(self.model.intercept_)
        coefficients = dict(zip(self.predictor_columns, self.model.coef_))

        Logger.alert(f"Beta0 (intercepto) = {intercept:.4f}", LogType.INFO)

        for name, coef in coefficients.items():
            Logger.alert(
                f"Beta '{name}' = {coef:.4f} -> mantendo as demais variáveis "
                f"constantes (Ceteris Paribus), cada unidade adicional em "
                f"'{name}' está associada a uma variação de {coef:.4f} unidade(s) "
                f"em '{self.response_column}'.",
                LogType.INFO,
            )

        return {"intercept": intercept, "coefficients": coefficients}

    def run_full_analysis(self):

        self.fit()
        self.evaluate()
        report = self.report()

        return {
            "response_column": self.response_column,
            "predictor_columns": self.predictor_columns,
            "n": len(self.y),
            "intercept": report["intercept"],
            "coefficients": report["coefficients"],
            "r2": self.r2,
            "rmse": self.rmse,
        }


if __name__ == "__main__":
    # Execução standalone: espera ser rodado a partir da pasta `src`
    # (mesma convenção do main.py: `cd src && python models/regression.py`)
    from extract.extractor import DataExtractor

    extractor = DataExtractor("../dados_limpos_final.csv")
    df = extractor.extract_csv()

    if df is None:
        Logger.alert(
            "Não foi possível carregar 'dados_limpos_final.csv'. Execute main.py primeiro.",
            LogType.WARNING,
        )
    else:
        analyzer = RegressionAnalyzer(
            dataframe=df,
            response_column="total_vaccinations",
            predictor_columns=["daily_vaccinations", "people_fully_vaccinated"],
        )

        results = analyzer.run_full_analysis()

        Logger.alert(f"Resumo Regressão: {results}", LogType.INFO)
