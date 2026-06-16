from extract.extractor import DataExtractor
from transform.transformer import DataTransformer
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

    final_df = transformer.get_dataframe()

    final_df.to_csv(
        "../data/processed/country_vaccinations_processed.csv",
        index=False
    )

    Logger.alert("Pipeline Finished with Success", log_type=LogType.SUCCESS)


if __name__ == "__main__":
    main()