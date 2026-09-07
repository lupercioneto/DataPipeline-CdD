import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logs.log import Logger, LogType


class BootstrapAnalyzer:
    
    # Valor crítico de z para 95% de confiança (aproximação normal)
    Z_95 = 1.96

    def __init__(
        self,
        dataframe: pd.DataFrame,
        column: str,
        n_replicas: int = 2000,
        random_state: int | None = 42,
    ):
        self.column = column
        # O enunciado exige no mínimo 2.000 réplicas
        self.n_replicas = max(n_replicas, 2000)
        self.random_state = random_state

        self.sample = (
            dataframe[column]
            .dropna()
            .astype(float)
            .to_numpy()
        )
        self.sample_size = len(self.sample)

        self.sample_mean = None
        self.sample_std = None
        self.bootstrap_means = None
        self.ci_bootstrap = None
        self.ci_parametric = None

    def compute_sample_statistics(self):
       

        Logger.alert(f"Calculando estatísticas amostrais de '{self.column}'...", LogType.INFO)

        if self.sample_size == 0:
            Logger.alert(f"Erro: coluna '{self.column}' não possui valores válidos!", LogType.ERROR)
            raise ValueError(f"Coluna '{self.column}' sem dados válidos para bootstrap.")

        self.sample_mean = float(np.mean(self.sample))
        # ddof=1 -> desvio padrão amostral (divide por N-1)
        self.sample_std = float(np.std(self.sample, ddof=1))

        Logger.alert(
            f"N={self.sample_size} | Media amostral (X_barra)={self.sample_mean:.4f} | "
            f"Desvio padrao amostral (s)={self.sample_std:.4f}",
            LogType.SUCCESS
        )

        return self.sample_mean, self.sample_std

    def run_resampling(self):

        Logger.alert(
            f"Executando reamostragem Bootstrap com {self.n_replicas} réplicas (com reposição)...",
            LogType.INFO
        )

        rng = np.random.default_rng(self.random_state)

        bootstrap_means = np.empty(self.n_replicas)

        for i in range(self.n_replicas):
            resample = rng.choice(self.sample, size=self.sample_size, replace=True)
            bootstrap_means[i] = resample.mean()

        self.bootstrap_means = bootstrap_means

        Logger.alert("Distribuição empírica bootstrap da média gerada com sucesso!", LogType.SUCCESS)

        return self.bootstrap_means

    def compute_confidence_intervals(self):

        if self.bootstrap_means is None:
            raise RuntimeError("Execute run_resampling() antes de calcular os intervalos de confiança.")

        Logger.alert("Calculando Intervalos de Confiança (95%)...", LogType.INFO)

        # Método Não-Paramétrico (Bootstrap): percentis 2.5% e 97.5%
        lower_bootstrap = float(np.percentile(self.bootstrap_means, 2.5))
        upper_bootstrap = float(np.percentile(self.bootstrap_means, 97.5))
        self.ci_bootstrap = (lower_bootstrap, upper_bootstrap)

        # Método Paramétrico Tradicional: X̄ ± z95% * (s / sqrt(N))
        standard_error = self.sample_std / np.sqrt(self.sample_size)
        margin = self.Z_95 * standard_error
        self.ci_parametric = (self.sample_mean - margin, self.sample_mean + margin)

        Logger.alert(
            f"IC 95% Bootstrap (percentil): [{lower_bootstrap:.4f}, {upper_bootstrap:.4f}]",
            LogType.SUCCESS
        )
        Logger.alert(
            f"IC 95% Paramétrico (normal): [{self.ci_parametric[0]:.4f}, {self.ci_parametric[1]:.4f}]",
            LogType.SUCCESS
        )

        return self.ci_bootstrap, self.ci_parametric

    def plot_distribution(self, output_path: str = "../distribuicao_bootstrap.png"):

        if self.bootstrap_means is None or self.ci_bootstrap is None or self.ci_parametric is None:
            raise RuntimeError("Execute run_resampling() e compute_confidence_intervals() antes de plotar.")

        Logger.alert("Gerando gráfico de distribuição bootstrap...", LogType.INFO)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            self.bootstrap_means,
            bins=50,
            color="#1f77b4",
            alpha=0.75,
            edgecolor="white"
        )

        # Limites do IC Bootstrap (não-paramétrico)
        ax.axvline(
            self.ci_bootstrap[0], color="#d62728", linestyle="--", linewidth=2,
            label="IC 95% Bootstrap (percentil)"
        )
        ax.axvline(self.ci_bootstrap[1], color="#d62728", linestyle="--", linewidth=2)

        # Limites do IC Paramétrico (aproximação normal)
        ax.axvline(
            self.ci_parametric[0], color="#2ca02c", linestyle=":", linewidth=2,
            label="IC 95% Paramétrico (normal)"
        )
        ax.axvline(self.ci_parametric[1], color="#2ca02c", linestyle=":", linewidth=2)

        ax.axvline(
            self.sample_mean, color="black", linewidth=1.5,
            label=f"Média amostral (X̄ = {self.sample_mean:.2f})"
        )

        ax.set_xlabel(f"Média Bootstrap de '{self.column}'")
        ax.set_ylabel(f"Frequência (em {self.n_replicas} réplicas)")
        ax.set_title(
            "Distribuição Empírica Bootstrap da Média\ne Intervalos de Confiança (95%)",
            fontsize=13,
            fontweight="bold"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        Logger.alert(f"Gráfico salvo com sucesso em: {output_path}", LogType.SUCCESS)

    def run_full_analysis(self, output_path: str = "../distribuicao_bootstrap.png"):

        self.compute_sample_statistics()
        self.run_resampling()
        self.compute_confidence_intervals()
        self.plot_distribution(output_path)

        return {
            "column": self.column,
            "n": self.sample_size,
            "n_replicas": self.n_replicas,
            "sample_mean": self.sample_mean,
            "sample_std": self.sample_std,
            "ci_bootstrap_95": self.ci_bootstrap,
            "ci_parametric_95": self.ci_parametric,
        }


if __name__ == "__main__":
    from extract.extractor import DataExtractor

    extractor = DataExtractor("../dados_limpos_final.csv")
    df = extractor.extract_csv()

    if df is None:
        Logger.alert(
            "Não foi possível carregar 'dados_limpos_final.csv'. Execute main.py primeiro.",
            LogType.WARNING
        )
    else:
        analyzer = BootstrapAnalyzer(
            dataframe=df,
            column="people_vaccinated_per_hundred",
            n_replicas=2000,
            random_state=42,
        )

        results = analyzer.run_full_analysis(output_path="../distribuicao_bootstrap.png")

        Logger.alert(f"Resumo Bootstrap: {results}", LogType.INFO)
