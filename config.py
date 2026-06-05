from pathlib import Path

# General project settings
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sample_en.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
HF_CACHE_DIR = BASE_DIR / ".hf_cache"

WINDOW_SIZE = 10
TOP_N_WORDS = 10

# Language-specific preprocessing settings
LANGUAGE_MODE = "en"  # "auto", "en", "ru"
EN_ALLOWED_POS = {"NOUN", "ADJ", "PROPN"}
RU_ALLOWED_POS = {"NOUN", "ADJ", "PROPN"}
MIN_TOKEN_LENGTH = 3

# Graph mode: "baseline" = nearby words, "semantic" = embedding similarity
GRAPH_MODE = "semantic"

# Baseline/co-occurrence graph
COOCCURRENCE_MIN_WEIGHT = 1

# Semantic graph
MIN_WORD_FREQ = 2
MAX_UNIQUE_WORDS = 70
SEMANTIC_TOP_K = 2

AUTO_THRESHOLD_PERCENTILE = 90.0
SEMANTIC_MIN_SIM = None

# Visualization settings
SAVE_PLOTS = True
SHOW_PLOTS = False
