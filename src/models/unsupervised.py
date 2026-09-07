import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from logs.log import Logger, LogType


class UnsupervisedAnalyzer:
    """
    Redução de dimensionalidade via PCA (fundamentada em SVD) e
    clusterização via K-Means, com seleção do número de clusters (k)
    pelo Método do Cotovelo.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_columns: list[str],
        n_clusters: int = 3,
        k_range: range = range(1, 11),
        random_state: int = 42,
    ):
        self.feature_columns = feature_columns
        self.n_clusters = n_clusters
        self.k_range = k_range
        self.random_state = random_state

        self.X = dataframe[feature_columns].fillna(0)
        self.X_scaled = None

        self.pca = None
        self.X_pca = None
        self.explained_variance_ratio = None

        self.inertias = None
        self.kmeans = None
        self.cluster_labels = None

    def prepare_features(self):
        """Padroniza (StandardScaler) as variáveis numéricas selecionadas."""

        Logger.alert(
            f"Padronizando variáveis numéricas: {', '.join(self.feature_columns)}",
            LogType.INFO,
        )

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        Logger.alert("Padronização concluída!", LogType.SUCCESS)

        return self.X_scaled

    def run_pca(self, n_components: int = 2):
        """Aplica PCA (fundamentado em SVD) e projeta as observações no
        plano dos dois primeiros componentes principais."""

        if self.X_scaled is None:
            raise RuntimeError("Execute prepare_features() antes do PCA.")

        Logger.alert(f"Aplicando PCA (n_components={n_components})...", LogType.INFO)

        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        Logger.alert(
            f"Variância explicada: PC1={self.explained_variance_ratio[0]:.4f} | "
            f"PC2={self.explained_variance_ratio[1]:.4f} | "
            f"Acumulada={self.explained_variance_ratio.sum():.4f}",
            LogType.SUCCESS,
        )

        return self.X_pca, self.explained_variance_ratio

    def plot_pca_projection(self, output_path: str = "../plots/pca_projecao.png"):
        """Salva o gráfico de dispersão da projeção PCA (sem coloração por
        cluster), conforme exigido pelo enunciado."""

        if self.X_pca is None:
            raise RuntimeError("Execute run_pca() antes de plotar a projeção.")

        Logger.alert("Gerando gráfico de projeção PCA...", LogType.INFO)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            self.X_pca[:, 0], self.X_pca[:, 1],
            alpha=0.6, edgecolors="w", color="#1f77b4",
        )
        ax.set_title("Projeção PCA das Observações (PC1 x PC2)", fontsize=13, fontweight="bold")
        ax.set_xlabel(
            f"Componente Principal 1 ({self.explained_variance_ratio[0]*100:.1f}% da variância)"
        )
        ax.set_ylabel(
            f"Componente Principal 2 ({self.explained_variance_ratio[1]*100:.1f}% da variância)"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        Logger.alert(f"Gráfico salvo com sucesso em: {output_path}", LogType.SUCCESS)

    def run_elbow_method(self, output_path: str = "../plots/curva_cotovelo_kmeans.png"):
        """Plota a inércia contra k para determinar o número ótimo de
        clusters (Método do Cotovelo)."""

        if self.X_scaled is None:
            raise RuntimeError("Execute prepare_features() antes do método do cotovelo.")

        Logger.alert("Executando Método do Cotovelo (Elbow Method)...", LogType.INFO)

        self.inertias = []
        for k in self.k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.X_scaled)
            self.inertias.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(self.k_range), self.inertias, marker="o")
        ax.set_title("Método do Cotovelo (Elbow Method)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Número de Clusters (k)")
        ax.set_ylabel("Inércia")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        Logger.alert(f"Gráfico salvo com sucesso em: {output_path}", LogType.SUCCESS)

        return self.inertias

    def run_kmeans(self):
        """Ajusta o K-Means final com o número de clusters escolhido
        (n_clusters) a partir do cotovelo."""

        if self.X_scaled is None:
            raise RuntimeError("Execute prepare_features() antes do K-Means.")

        Logger.alert(f"Ajustando K-Means com k={self.n_clusters}...", LogType.INFO)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)

        Logger.alert("Clusterização K-Means concluída!", LogType.SUCCESS)

        return self.cluster_labels

    def plot_clusters(self, output_path: str = "../plots/clusters_kmeans.png"):
        """Salva a projeção PCA identificando visualmente cada cluster
        por cores distintas."""

        if self.X_pca is None or self.cluster_labels is None:
            raise RuntimeError("Execute run_pca() e run_kmeans() antes de plotar os clusters.")

        Logger.alert("Gerando gráfico de clusters projetados no PCA...", LogType.INFO)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            self.X_pca[:, 0], self.X_pca[:, 1],
            c=self.cluster_labels, cmap="viridis", alpha=0.6, edgecolors="w",
        )
        ax.set_title(
            f"Clusters K-Means (k={self.n_clusters}) Projetados no PCA",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        fig.colorbar(scatter, label="Cluster")
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        Logger.alert(f"Gráfico salvo com sucesso em: {output_path}", LogType.SUCCESS)

    def run_full_analysis(
        self,
        pca_output_path: str = "../plots/pca_projecao.png",
        elbow_output_path: str = "../plots/curva_cotovelo_kmeans.png",
        clusters_output_path: str = "../plots/clusters_kmeans.png",
    ):

        self.prepare_features()
        self.run_pca()
        self.plot_pca_projection(pca_output_path)
        self.run_elbow_method(elbow_output_path)
        self.run_kmeans()
        self.plot_clusters(clusters_output_path)

        return {
            "feature_columns": self.feature_columns,
            "explained_variance_ratio": tuple(self.explained_variance_ratio),
            "explained_variance_cumulative": float(self.explained_variance_ratio.sum()),
            "n_clusters": self.n_clusters,
            "inertias": self.inertias,
            "cluster_sizes": pd.Series(self.cluster_labels).value_counts().to_dict(),
        }


if __name__ == "__main__":
    # Execução standalone: espera ser rodado a partir da pasta `src`
    # (mesma convenção do main.py: `cd src && python models/unsupervised.py`)
    from extract.extractor import DataExtractor

    extractor = DataExtractor("../dados_limpos_final.csv")
    df = extractor.extract_csv()

    if df is None:
        Logger.alert(
            "Não foi possível carregar 'dados_limpos_final.csv'. Execute main.py primeiro.",
            LogType.WARNING,
        )
    else:
        analyzer = UnsupervisedAnalyzer(
            dataframe=df,
            feature_columns=[
                "total_vaccinations_per_hundred",
                "people_vaccinated_per_hundred",
                "people_fully_vaccinated_per_hundred",
                "daily_vaccinations_per_million",
            ],
            n_clusters=3,
        )

        results = analyzer.run_full_analysis()

        Logger.alert(f"Resumo Não Supervisionado: {results}", LogType.INFO)
