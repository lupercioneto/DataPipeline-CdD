from extract.extractor import DataExtractor
from transform.transformer import DataTransformer
from visualize.visualize import DataVisualizer
from inference.bootstrap import BootstrapAnalyzer
from inference.ab_testing import ABTestAnalyzer
from models.regression import RegressionAnalyzer
from models.machine_learning import ClassificationAnalyzer
from models.unsupervised import UnsupervisedAnalyzer
from logs.log import Logger, LogType


def main():

    extractor = DataExtractor(
        "../data/original/country_vaccinations.csv"
    )

    df = extractor.extract_csv()

    if df is None:
        Logger.alert(message="Pipeline Finished", log_type=LogType.WARNING)
        return

    transformer = DataTransformer(df)

    transformer.standardize_strings()
    transformer.normalize_dates()
    transformer.convert_numeric_columns()
    transformer.handle_nulls()
    transformer.remove_duplicates()
    transformer.remove_outliers_iqr("daily_vaccinations")

    final_df = transformer.get_dataframe()

    # Arquivo final consolidado, conforme exigido no enunciado (raiz do projeto)
    final_df.to_csv(
        "../dados_limpos_final.csv",
        index=False
    )

    visualizer = DataVisualizer(final_df)
    visualizer.plot_daily_vaccinations_evolution(
        output_path="../plots/dados_vacinacao_diaria.png"
    )

    # --- Parte 2: Inferência Estatística, Testes A/B e Causalidade ---

    # 4.1 Estimação de Parâmetros e Bootstrap
    bootstrap_analyzer = BootstrapAnalyzer(
        dataframe=final_df,
        column="people_vaccinated_per_hundred",
        n_replicas=2000,
        random_state=42,
    )
    bootstrap_results = bootstrap_analyzer.run_full_analysis(
        output_path="../plots/distribuicao_bootstrap.png"
    )
    Logger.alert(f"Resumo Bootstrap: {bootstrap_results}", LogType.INFO)

    # 4.2 Teste de Hipóteses e Teste A/B (Alta vs. Baixa cobertura vacinal)
    segmented_df = ABTestAnalyzer.split_into_groups(
        final_df,
        split_column="people_vaccinated_per_hundred",
        group_column="cobertura_grupo",
    )
    ab_test_analyzer = ABTestAnalyzer(
        dataframe=segmented_df,
        metric_column="daily_vaccinations",
        group_column="cobertura_grupo",
        n_permutations=2000,
        random_state=42,
    )
    ab_test_results = ab_test_analyzer.run_full_analysis(
        output_path="../plots/distribuicao_permutacao.png"
    )
    Logger.alert(f"Resumo Teste A/B: {ab_test_results}", LogType.INFO)

    # 4.3 Modelagem Preditiva Supervisionada (Regressão Múltipla)
    regression_analyzer = RegressionAnalyzer(
        dataframe=final_df,
        response_column="total_vaccinations",
        predictor_columns=["daily_vaccinations", "people_fully_vaccinated"],
    )
    regression_results = regression_analyzer.run_full_analysis()
    Logger.alert(f"Resumo Regressão: {regression_results}", LogType.INFO)

    # 4.3 Modelagem Preditiva Supervisionada (Classificação: Reg. Logística vs. KNN)
    classification_analyzer = ClassificationAnalyzer(
        dataframe=final_df,
        feature_columns=[
            "total_vaccinations_per_hundred",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
        ],
        target_source_column="daily_vaccinations_per_million",
        target_column="is_fast_vax",
    )
    classification_results = classification_analyzer.run_full_analysis()
    Logger.alert(f"Resumo Classificação: {classification_results}", LogType.INFO)

    # 4.4 Aprendizado Não Supervisionado (PCA + K-Means)
    unsupervised_analyzer = UnsupervisedAnalyzer(
        dataframe=final_df,
        feature_columns=[
            "total_vaccinations_per_hundred",
            "people_vaccinated_per_hundred",
            "people_fully_vaccinated_per_hundred",
            "daily_vaccinations_per_million",
        ],
        n_clusters=3,
    )
    unsupervised_results = unsupervised_analyzer.run_full_analysis(
        pca_output_path="../plots/pca_projecao.png",
        elbow_output_path="../plots/curva_cotovelo_kmeans.png",
        clusters_output_path="../plots/clusters_kmeans.png",
    )
    Logger.alert(f"Resumo Não Supervisionado: {unsupervised_results}", LogType.INFO)

    Logger.alert("Pipeline Finished with Success", log_type=LogType.SUCCESS)


if __name__ == "__main__":
    main()