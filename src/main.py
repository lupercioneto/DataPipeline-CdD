from extract.extractor import DataExtractor
from transform.transformer import DataTransformer
from logs.log import Logger, LogType


def main():

    extractor = DataExtractor(
        "../data/original/country_vaccinations.csv"
    )

    df = extractor.extract_csv()
    columns = [
        "daily_vaccinations",
        "people_vaccinated",
        "people_fully_vaccinated"
    ]

    if df is None:
        Logger.alert(message="Pipeline Finished", log_type=LogType.WARNING)
        return

    transformer = DataTransformer(df)

    transformer.standardize_strings()
    transformer.normalize_dates()
    transformer.convert_numeric_columns()
    transformer.handle_nulls()
    transformer.remove_duplicates()
    
    # Remove só uma coluna
    # transformer.remove_outliers_iqr("daily_vaccinations")

    # Remove da lista columns
    """for col in columns:
        transformer.remove_outliers_iqr(col)"""
    
    final_df = transformer.get_dataframe()

    final_df.to_csv(
        "../data/processed/country_vaccinations_processed.csv",
        index=False
    )

    Logger.alert("Pipeline Finished with Success", log_type=LogType.SUCCESS)


if __name__ == "__main__":
    main()