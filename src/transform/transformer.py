import pandas as pd
from unidecode import unidecode
from logs.log import Logger, LogType


class DataTransformer:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def standardize_strings(self):

        Logger.alert("Initiating Strings Normalize!", log_type=LogType.INFO)

        string_columns = self.df.select_dtypes(include="object").columns

        for col in string_columns:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .apply(unidecode)
            )

        Logger.alert("Strings Normalized!", log_type=LogType.SUCCESS)

    def normalize_dates(self):

        Logger.alert("Normalizing Dates!", log_type=LogType.INFO)

        if "date" in self.df.columns:

            self.df["date"] = pd.to_datetime(
                self.df["date"],
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        Logger.alert("Dates Normalized!", log_type=LogType.SUCCESS)

    def convert_numeric_columns(self):

        Logger.alert("Converting Numeric Columns!", log_type=LogType.INFO)

        numeric_columns = [
            "total_vaccinations",
            "people_vaccinated",
            "people_fully_vaccinated",
            "daily_vaccinations"
        ]

        for col in numeric_columns:

            if col in self.df.columns:

                self.df[col] = pd.to_numeric(
                    self.df[col],
                    errors="coerce"
                )

        Logger.alert("Numeric Conversion Accomplished!", log_type=LogType.SUCCESS)

    def remove_outliers_iqr(self, column):

        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        original_size = len(self.df)

        self.df = self.df[
            (self.df[column] >= lower) &
            (self.df[column] <= upper)
        ]

        removed = original_size - len(self.df)

        Logger.alert(
            f"{removed} outliers removed from column {column}", log_type=LogType.SUCCESS)

    def handle_nulls(self):

        Logger.alert("Processing null values!", log_type=LogType.INFO)

        # Estratégia:
        # Dados críticos ausentes serão removidos

        self.df.dropna(
            subset=["country", "date"],
            inplace=True
        )

        # Campos numéricos recebem 0
        # porque ausência pode indicar falta de registro

        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(0)
        Logger.alert("Null Values Processed!", log_type=LogType.SUCCESS)

    def remove_duplicates(self):

        Logger.alert("Removing Duplicates!", log_type=LogType.INFO)
        self.df.drop_duplicates(inplace=True)
        Logger.alert("Removed Duplicates!", log_type=LogType.SUCCESS)

    def get_dataframe(self):

        return self.df