import os
from collections import Counter
from functools import lru_cache

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from config import HF_CACHE_DIR, MIN_TOKEN_LENGTH

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer


MODEL_NAMES = {
    "en": "all-MiniLM-L6-v2",
    "ru": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def _prepare_cache_dir():
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@lru_cache(maxsize=2)
def _get_embedding_model(language):
    _prepare_cache_dir()
    model_name = MODEL_NAMES.get(language)
    if model_name is None:
        raise ValueError(f"Unsupported language: {language}")
    return SentenceTransformer(model_name, cache_folder=str(HF_CACHE_DIR))


def _select_vocabulary(words, min_freq=2, max_words=200):
    freq = Counter(words)
    selected = [
        word
        for word, count in freq.most_common()
        if count >= min_freq and len(word) >= MIN_TOKEN_LENGTH
    ]
    return selected[:max_words]


def get_word_embeddings(words, language, min_freq=2, max_words=80):
    selected_words = _select_vocabulary(words, min_freq, max_words)
    if not selected_words:
        return selected_words, []

    model = _get_embedding_model(language)
    embeddings = model.encode(selected_words)
    return selected_words, embeddings


def build_semantic_graph(
    words,
    language,
    top_k=4,
    min_similarity=0.20,
    min_freq=2,
    max_words=200,
):
    unique_words = _select_vocabulary(words, min_freq=min_freq, max_words=max_words)

    G = nx.Graph()
    for word in unique_words:
        G.add_node(word)

    if len(unique_words) < 2:
        return G

    model = _get_embedding_model(language)
    embeddings = model.encode(unique_words)
    sim_matrix = cosine_similarity(embeddings)

    for i, word in enumerate(unique_words):
        neighbors = sorted(
            enumerate(sim_matrix[i]),
            key=lambda item: item[1],
            reverse=True,
        )

        for j, sim in neighbors[1 : top_k + 1]:
            if sim >= min_similarity:
                G.add_edge(word, unique_words[j], weight=float(sim))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G
