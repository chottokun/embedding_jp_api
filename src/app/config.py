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
    with open(MODELS_FILE, 'r') as f:
        data = yaml.safe_load(f)
        if data:
            SUPPORTED_MODELS = data

EMBEDDING_MODELS = SUPPORTED_MODELS.get("embedding_models", [])
RERANK_MODELS = SUPPORTED_MODELS.get("rerank_models", [])
