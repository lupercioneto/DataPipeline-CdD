
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logs.log import Logger, LogType


class ABTestAnalyzer:
   
    ALPHA = 0.05  # Limite de Significância padrão

    def __init__(
        self,
        dataframe: pd.DataFrame,
        metric_column: str,
        group_column: str,
        n_permutations: int = 2000,
        random_state: int | None = 42,
    ):
        self.dataframe = dataframe
        self.metric_column = metric_column
        self.group_column = group_column
        # O enunciado exige no mínimo 2.000 iterações
        self.n_permutations = max(n_permutations, 2000)
        self.random_state = random_state

        self.group_a = None
        self.group_b = None
        self.observed_diff = None
        self.permutation_diffs = None
        self.p_value = None

    @staticmethod
    def split_into_groups(
        dataframe: pd.DataFrame,
        split_column: str,
        group_column: str = "cobertura_grupo",
        threshold: float | None = None,
    ) -> pd.DataFrame:
        
        df = dataframe.copy()
        cutoff = threshold if threshold is not None else df[split_column].median()

        df[group_column] = np.where(df[split_column] >= cutoff, "alta", "baixa")

        Logger.alert(
            f"Grupos definidos por '{split_column}' (corte={cutoff:.4f}): "
            f"Grupo A = 'alta' (>= corte), Grupo B = 'baixa' (< corte)",
            LogType.INFO
        )

        return df

    def prepare_groups(self):
        """Isola os vetores numéricos da métrica de interesse para os
        Grupos A e B, descartando linhas com valores ausentes."""

        Logger.alert("Preparando Grupo A (alta cobertura) e Grupo B (baixa cobertura)...", LogType.INFO)

        clean_df = self.dataframe[[self.group_column, self.metric_column]].dropna()

        self.group_a = clean_df.loc[
            clean_df[self.group_column] == "alta", self.metric_column
        ].to_numpy(dtype=float)

        self.group_b = clean_df.loc[
            clean_df[self.group_column] == "baixa", self.metric_column
        ].to_numpy(dtype=float)

        if len(self.group_a) == 0 or len(self.group_b) == 0:
            Logger.alert("Erro: um dos grupos ficou vazio após a segmentação!", LogType.ERROR)
            raise ValueError("Grupos A e B precisam conter observações válidas.")

        Logger.alert(
            f"Grupo A: n={len(self.group_a)}, media_A={self.group_a.mean():.4f} | "
            f"Grupo B: n={len(self.group_b)}, media_B={self.group_b.mean():.4f}",
            LogType.SUCCESS
        )

        return self.group_a, self.group_b

    def compute_observed_statistic(self):
        """Calcula a estatística de teste observada: diferença entre as
        médias amostrais dos dois grupos (X̄_A - X̄_B)."""

        if self.group_a is None or self.group_b is None:
            raise RuntimeError("Execute prepare_groups() antes de calcular a estatística observada.")

        self.observed_diff = float(self.group_a.mean() - self.group_b.mean())

        Logger.alert(
            f"Estatistica de teste observada (media_A - media_B) = {self.observed_diff:.4f}",
            LogType.SUCCESS
        )

        return self.observed_diff

    def run_permutation_test(self):
        """Implementa o Teste de Permutação (embaralhamento de rótulos) para
        simular a distribuição da estatística sob a validade de H0."""

        if self.observed_diff is None:
            raise RuntimeError("Execute compute_observed_statistic() antes do teste de permutação.")

        Logger.alert(
            f"Executando Teste de Permutação com {self.n_permutations} iterações (H0: mu_A = mu_B)...",
            LogType.INFO
        )

        rng = np.random.default_rng(self.random_state)

        pooled = np.concatenate([self.group_a, self.group_b])
        n_a = len(self.group_a)

        permutation_diffs = np.empty(self.n_permutations)

        for i in range(self.n_permutations):
            shuffled = rng.permutation(pooled)
            permutation_diffs[i] = shuffled[:n_a].mean() - shuffled[n_a:].mean()

        self.permutation_diffs = permutation_diffs

        # Valor-p empírico bicaudal: proporção de permutações tão ou mais
        # extremas que a diferença observada (correção +1 evita p=0 e produz
        # um estimador não enviesado do p-valor real).
        extreme_count = np.sum(np.abs(permutation_diffs) >= np.abs(self.observed_diff))
        self.p_value = (extreme_count + 1) / (self.n_permutations + 1)

        Logger.alert(f"Valor-p empírico (bicaudal) = {self.p_value:.5f}", LogType.SUCCESS)

        return self.permutation_diffs, self.p_value

    def evaluate_hypothesis(self):
        """Conclui formalmente se H0 deve ser rejeitada ao nível de
        significância de 5%."""

        if self.p_value is None:
            raise RuntimeError("Execute run_permutation_test() antes de avaliar a hipótese.")

        reject_h0 = self.p_value < self.ALPHA

        if reject_h0:
            conclusion = (
                f"p-valor ({self.p_value:.5f}) < alpha ({self.ALPHA}): rejeita-se H0. "
                "Há evidência estatística de diferença no ritmo de vacinação entre os grupos."
            )
            Logger.alert(conclusion, LogType.SUCCESS)
        else:
            conclusion = (
                f"p-valor ({self.p_value:.5f}) >= alpha ({self.ALPHA}): não se rejeita H0. "
                "Não há evidência estatística suficiente de diferença entre os grupos."
            )
            Logger.alert(conclusion, LogType.WARNING)

        return reject_h0, conclusion

    def plot_null_distribution(self, output_path: str = "../distribuicao_permutacao.png"):
        

        if self.permutation_diffs is None or self.observed_diff is None:
            raise RuntimeError("Execute run_permutation_test() antes de plotar.")

        Logger.alert("Gerando gráfico da distribuição sob H0 (teste de permutação)...", LogType.INFO)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            self.permutation_diffs,
            bins=50,
            color="#7f7f7f",
            alpha=0.75,
            edgecolor="white",
            label="Distribuição sob H0 (rótulos embaralhados)"
        )

        ax.axvline(
            self.observed_diff, color="#d62728", linewidth=2.5,
            label=f"Diferença observada (X̄_A - X̄_B = {self.observed_diff:.4f})"
        )
        # Referência simétrica, útil para visualizar o teste bicaudal
        ax.axvline(-self.observed_diff, color="#d62728", linewidth=1.5, linestyle="--")

        ax.set_xlabel(f"Diferença de médias simulada ({self.metric_column}: A - B)")
        ax.set_ylabel(f"Frequência (em {self.n_permutations} permutações)")
        ax.set_title(
            f"Teste de Permutação sob H0 (p-valor bicaudal = {self.p_value:.5f})",
            fontsize=13,
            fontweight="bold"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        Logger.alert(f"Gráfico salvo com sucesso em: {output_path}", LogType.SUCCESS)

    def run_full_analysis(self, output_path: str = "../distribuicao_permutacao.png"):
        

        self.prepare_groups()
        self.compute_observed_statistic()
        self.run_permutation_test()
        reject_h0, conclusion = self.evaluate_hypothesis()
        self.plot_null_distribution(output_path)

        return {
            "metric_column": self.metric_column,
            "group_a_n": len(self.group_a),
            "group_b_n": len(self.group_b),
            "observed_diff": self.observed_diff,
            "p_value": self.p_value,
            "alpha": self.ALPHA,
            "reject_h0": reject_h0,
            "conclusion": conclusion,
        }


if __name__ == "__main__":
    # Execução standalone: espera ser rodado a partir da pasta `src`
    # (mesma convenção do main.py: `cd src && python inference/ab_testing.py`)
    from extract.extractor import DataExtractor

    extractor = DataExtractor("../dados_limpos_final.csv")
    df = extractor.extract_csv()

    if df is None:
        Logger.alert(
            "Não foi possível carregar 'dados_limpos_final.csv'. Execute main.py primeiro.",
            LogType.WARNING
        )
    else:
        segmented_df = ABTestAnalyzer.split_into_groups(
            df,
            split_column="people_vaccinated_per_hundred",
            group_column="cobertura_grupo",
        )

        analyzer = ABTestAnalyzer(
            dataframe=segmented_df,
            metric_column="daily_vaccinations",
            group_column="cobertura_grupo",
            n_permutations=2000,
            random_state=42,
        )

        results = analyzer.run_full_analysis(output_path="../distribuicao_permutacao.png")

        Logger.alert(f"Resumo Teste A/B: {results}", LogType.INFO)
