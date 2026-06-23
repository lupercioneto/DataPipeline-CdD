from extract.extractor import DataExtractor
from transform.transformer import DataTransformer
from visualize.visualize import DataVisualizer
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
        output_path="../dados_vacinacao_diaria.png"
    )

    Logger.alert("Pipeline Finished with Success", log_type=LogType.SUCCESS)


if __name__ == "__main__":
    main()