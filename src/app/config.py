import logging
import os
from pathlib import Path
import yaml


# --- Load .env File ---
def _load_env_file():
    """Load key-value pairs from .env file if it exists, without overriding existing OS environment variables."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception as e:
            logging.warning(f"Failed to load .env file: {e}")


_load_env_file()

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

# --- Logit Gate Configuration ---
LOGIT_GATE_MODELS = SUPPORTED_MODELS.get(
    "logit_gate_models", ["Qwen/Qwen2.5-1.5B-Instruct"]
)
LOGIT_GATE_ENABLED = os.getenv("LOGIT_GATE_ENABLED", "true").lower() in {
    "true",
    "1",
    "yes",
}
LOGIT_GATE_THRESHOLD = float(os.getenv("LOGIT_GATE_THRESHOLD", "0.55"))
LOGIT_GATE_ASCII_BOOST_WEIGHT = float(os.getenv("LOGIT_GATE_ASCII_BOOST_WEIGHT", "1.2"))
LOGIT_GATE_ASCII_MIN_TOKEN_LEN = int(os.getenv("LOGIT_GATE_ASCII_MIN_TOKEN_LEN", "3"))
LOGIT_GATE_MAX_DOC_CHARS = int(os.getenv("LOGIT_GATE_MAX_DOC_CHARS", "1500"))
LOGIT_GATE_BATCH_SIZE = int(os.getenv("LOGIT_GATE_BATCH_SIZE", "8"))
LOGIT_GATE_POS_TOKENS = [
    t.strip()
    for t in os.getenv("LOGIT_GATE_POS_TOKENS", "Yes,yes,はい").split(",")
    if t.strip()
]
LOGIT_GATE_NEG_TOKENS = [
    t.strip()
    for t in os.getenv("LOGIT_GATE_NEG_TOKENS", "No,no,いいえ").split(",")
    if t.strip()
]

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
# MAX_INPUT_LENGTH is set to 65536 characters.
# The model supports up to 8192 tokens.
# 65536 chars provides a safe margin (approx 8 chars/token) to cover the context window
# while preventing excessive resource consumption during tokenization.
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "65536"))

# MAX_INPUT_ITEMS is set to 256.
# Processing too many items in a single request can lead to timeouts and resource exhaustion (DoS).
# Clients should batch requests if they need to process more items.
MAX_INPUT_ITEMS = int(os.getenv("MAX_INPUT_ITEMS", "256"))

# API Key for authentication. If not set, authentication is disabled.
API_KEY = os.getenv("API_KEY")


# Multiple API Keys & client-specific rate limits:
# Format in env: "key1,key2" or JSON: '{"key1": 120, "key2": 300}'
def _load_api_keys():
    raw = os.getenv("API_KEYS", "").strip()
    keys_map = {}
    if not raw:
        if API_KEY:
            keys_map[API_KEY] = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
        return keys_map
    if raw.startswith("{"):
        import json

        try:
            parsed = json.loads(raw)
            for k, limit in parsed.items():
                keys_map[str(k)] = int(limit)
            return keys_map
        except Exception as e:
            logging.warning(f"Failed to parse API_KEYS JSON: {e}")
    for item in raw.split(","):
        k = item.strip()
        if k:
            keys_map[k] = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    if API_KEY and API_KEY not in keys_map:
        keys_map[API_KEY] = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    return keys_map


API_KEYS_MAP = _load_api_keys()

# Rate limit configuration (requests per minute per client)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

# Maximum concurrent inference requests (semaphore limit to protect GPU/CPU from OOM/saturation)
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
INFERENCE_SEMAPHORE_TIMEOUT_SECONDS = float(
    os.getenv("INFERENCE_SEMAPHORE_TIMEOUT_SECONDS", "30.0")
)

# Graceful shutdown configuration (seconds to wait for in-flight requests to complete)
SHUTDOWN_DRAIN_TIMEOUT_SECONDS = float(
    os.getenv("SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "10.0")
)

# Inference precision configuration: "float16", "bfloat16", or "" (default/float32)
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "").lower()

# Preload models on application startup (comma-separated list of model names)
PRELOAD_MODELS = [
    m.strip() for m in os.getenv("PRELOAD_MODELS", "").split(",") if m.strip()
]

# --- TEI Integration Configuration ---
# If these environment variables are set, the API will proxy requests to TEI.
EMBEDDING_TEI_URL = os.getenv("EMBEDDING_TEI_URL")
RERANK_TEI_URL = os.getenv("RERANK_TEI_URL")

# --- Offline Mode Configuration ---
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() in {"true", "1", "yes"}
if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    logging.info("Offline mode is enabled. Hugging Face Hub access is disabled.")
else:
    logging.info(
        "Offline mode is disabled. Hugging Face Hub access is enabled if needed."
    )
