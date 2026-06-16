from datetime import datetime
from enum import Enum


class LogType(Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Logger:

    @staticmethod
    def alert(message: str, log_type: LogType):

        print(
            f"[{log_type.value}] "
            f"{datetime.now()} - {message}"
        )