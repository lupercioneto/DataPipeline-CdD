import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from logs.log import Logger, LogType


class DataVisualizer:
    """
    Camada de visualização científica do pipeline.

    Responsável por traduzir graficamente os dados já tratados
    (dados_limpos_final.csv), respeitando os princípios de
    integridade visual: eixos identificados, unidades explícitas,
    escala coerente (iniciando em zero) e ausência de distorções.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def plot_daily_vaccinations_evolution(
        self,
        output_path: str = "../dados_vacinacao_diaria.png"
    ):
        """
        Gráfico de Linhas: Evolução temporal da métrica principal
        (total de vacinações diárias aplicadas no mundo, agregando
        todos os países por data).
        """

        Logger.alert(
            message="Gerando gráfico de evolução temporal (linhas)...",
            log_type=LogType.INFO
        )

        if "date" not in self.df.columns or "daily_vaccinations" not in self.df.columns:
            Logger.alert(
                message="Erro: colunas 'date' ou 'daily_vaccinations' ausentes!",
                log_type=LogType.ERROR
            )
            return

        # Agregação: soma de vacinações diárias de todos os países, por data
        serie = (
            self.df.groupby("date")["daily_vaccinations"]
            .sum()
            .reset_index()
            .sort_values("date")
        )
        serie["date"] = pd.to_datetime(serie["date"])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            serie["date"],
            serie["daily_vaccinations"],
            color="#1f77b4",
            linewidth=1.5
        )

        # Integridade visual: eixo Y começando em zero
        ax.set_ylim(bottom=0)

        # Unidades explícitas nos eixos
        ax.set_xlabel("Data")
        ax.set_ylabel("Vacinações aplicadas por dia (unidades, soma global)")
        ax.set_title(
            "Evolução Temporal da Vacinação Diária contra COVID-19 (Mundo)",
            fontsize=13,
            fontweight="bold"
        )

        # Evita notação científica truncada no eixo Y, exibe números completos
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", "."))
        )

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        fig.tight_layout()

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        Logger.alert(
            message=f"Gráfico salvo com sucesso em: {output_path}",
            log_type=LogType.SUCCESS
        )