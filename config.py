# Общие настройки проекта
WINDOW_SIZE = 6
ALLOWED_POS = {"NOUN", "ADJ", "PROPN"}
TOP_N_WORDS = 8

# Режим графа: "baseline" = слова рядом, "semantic" = смысловая близость
GRAPH_MODE = "semantic"

# Baseline/co-occurrence graph
COOCCURRENCE_MIN_WEIGHT = 2

# Semantic graph
MIN_WORD_FREQ = 2
MAX_UNIQUE_WORDS = 180
SEMANTIC_TOP_K = 4
SEMANTIC_MIN_SIM = 0.20
