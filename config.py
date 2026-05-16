from pathlib import Path

# General project settings
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sample_ru.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
HF_CACHE_DIR = BASE_DIR / ".hf_cache"

WINDOW_SIZE = 6
TOP_N_WORDS = 8

# Language-specific preprocessing settings
LANGUAGE_MODE = "auto"  # "auto", "en", "ru"
EN_ALLOWED_POS = {"NOUN", "ADJ", "PROPN"}
RU_ALLOWED_POS = {"NOUN", "ADJ", "PROPN"}
MIN_TOKEN_LENGTH = 3

# Graph mode: "baseline" = nearby words, "semantic" = embedding similarity
GRAPH_MODE = "baseline"

# Baseline/co-occurrence graph
COOCCURRENCE_MIN_WEIGHT = 2

# Semantic graph
MIN_WORD_FREQ = 2
MAX_UNIQUE_WORDS = 50
SEMANTIC_TOP_K = 2
SEMANTIC_MIN_SIM = 0.50

# Visualization settings
SAVE_PLOTS = True
SHOW_PLOTS = False
