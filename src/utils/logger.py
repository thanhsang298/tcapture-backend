import logging
from typing import Any, Dict
from src.config import TaiyoConfig

class Logger:
    _instance = None

    def __new__(cls, config: TaiyoConfig):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize_logger(config)
        return cls._instance

    def _initialize_logger(self, config: TaiyoConfig):
        self.logger = logging.getLogger("TaiyoLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent log duplication

        # Check if handlers are already added to avoid duplicate logs
        if not self.logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(config.Log.PATH)
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Stream Handler (Console)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            )
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)

    def debug(self, message: str, **kwargs: Any):
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs: Any):
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs: Any):
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs: Any):
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message: str, **kwargs: Any):
        self.logger.critical(self._format_message(message, kwargs))

    def _format_message(self, message: str, kwargs: Dict[str, Any]) -> str:
        if kwargs:
            formatted_vars = ", ".join(f"{key}={value}" for key, value in kwargs.items())
            return f"{message} | {formatted_vars}"
        return message

