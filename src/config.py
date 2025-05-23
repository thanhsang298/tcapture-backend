import logging
from pydantic_settings import BaseSettings
from typing import Tuple

class TaiyoConfig(BaseSettings):
    class API:
        PROJECT_NAME: str = "Taiyo TCapture"
        API_V1_STR: str = "/api/v1"
        PORT: int = 8000
        IMG_HEIGHT: int = 3648
        IMG_WIDTH: int = 5472

    class Log:
        PATH: str = "app.log"
        LEVEL: int = logging.INFO

    class LogType:
        DEBUG = logging.DEBUG
        INFO = logging.INFO
        WARNING = logging.WARNING
        ERROR = logging.ERROR
        CRITICAL = logging.CRITICAL
        
    class Corners:
        # Crop coordinates in format (top, left, bottom, right)
        TL: Tuple[int, int, int, int] = (0, 0, 3472, 2648) # keep x1, y1
        TR: Tuple[int, int, int, int] = (2000, 0, 5472, 2648) # keep y1, x2
        BR: Tuple[int, int, int, int] = (2000, 1000, 5472, 3648) # keep x2, y2
        BL: Tuple[int, int, int, int] = (0, 1000, 3472, 3648) # keep x1, y2
        DEBUG: bool = False  
        VIS_PATH: str = "./visualization"