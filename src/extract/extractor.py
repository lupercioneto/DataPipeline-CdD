import pandas as pd
from logs.log import Logger, LogType

class DataExtractor:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_csv(self) -> pd.DataFrame | None:

        try:
            Logger.alert(message=f"Reading File: {self.file_path}", log_type=LogType.INFO)

            df = pd.read_csv(self.file_path)

            Logger.alert(message="File Success Loaded!", log_type=LogType.SUCCESS)

            return df

        except FileNotFoundError:
            Logger.alert(message="Error: File NOT FOUND!", log_type=LogType.ERROR)
            return None

        except pd.errors.EmptyDataError:
            Logger.alert(message="Error: EMPTY file!", log_type=LogType.ERROR)
            return None

        except pd.errors.ParserError:
            Logger.alert(message="Failed to interpret CSV File!", log_type=LogType.ERROR)
            return None

        except Exception as e:
            Logger.alert(message=f"Unexpected Error: {e}", log_type=LogType.ERROR)
            return None