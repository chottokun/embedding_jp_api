import os
import yaml
from pathlib import Path

# --- Port Configuration ---
# Read port from environment variable APP_PORT, with a default of 8000.
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# --- Model Configuration ---
# Load the list of supported models from the YAML file.
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
MODELS_FILE = CONFIG_DIR / "models.yml"

SUPPORTED_MODELS = {}
if MODELS_FILE.exists():
    with open(MODELS_FILE, "r") as f:
        data = yaml.safe_load(f)
        if data:
            SUPPORTED_MODELS = data

EMBEDDING_MODELS = SUPPORTED_MODELS.get("embedding_models", [])
RERANK_MODELS = SUPPORTED_MODELS.get("rerank_models", [])

# --- Ruri-v3 Prefix Mapping ---
RURI_PREFIX_MAP = {
    "query": "検索クエリ: ",
    "document": "検索文書: ",
    "classification": "トピック: ",
    "clustering": "トピック: ",
    "sts": "",
}

# --- Security Configuration ---
# Limits for input validation to prevent DoS attacks.
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "65536"))
MAX_INPUT_ITEMS = int(os.getenv("MAX_INPUT_ITEMS", "2048"))
