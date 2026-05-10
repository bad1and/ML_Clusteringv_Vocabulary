# Общие настройки проекта
ALLOWED_POS = {"NOUN", "ADJ", "PROPN"} # допустимые части речи
TOP_N_WORDS = 8 #

# Режим графа: "baseline" = слова рядом, "semantic" = смысловая близость
GRAPH_MODE = "semantic"

# Baseline/co-occurrence graph
COOCCURRENCE_MIN_WEIGHT = 2 # минимальный вес ребер
COOCCURRENCE_NEIGHBORS = 6 # максимальное кол-во соседей слова

# Semantic graph
MIN_WORD_FREQ = 2 # минимальное частота встречаемости слова
MAX_UNIQUE_WORDS = 50 # максимальный размер итогового массива слов
SEMANTIC_NEIGHBORS = 2 # максимальное кол-во соседей слова
SEMANTIC_MIN_SIM = 0.50 # минимальное косинусное сходство слов
