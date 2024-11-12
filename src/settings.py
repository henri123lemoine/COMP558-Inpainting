from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
DATASETS_PATH = DATA_PATH / "datasets"
MODELS_PATH = DATA_PATH / "models"
CACHE_PATH = DATA_PATH / ".cache"
LOGS_PATH = PROJECT_PATH / "logs"
LATEST_LOGS_FILE_PATH = LOGS_PATH / "latest.log"

for path in [DATASETS_PATH, MODELS_PATH, CACHE_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)
