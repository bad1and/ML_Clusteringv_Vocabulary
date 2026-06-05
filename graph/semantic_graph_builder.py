import os
from collections import Counter
from functools import lru_cache

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from config import HF_CACHE_DIR, MIN_TOKEN_LENGTH

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
from graph.similarity_threshold import get_similarity_threshold, get_similarity_stats

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


def _select_vocabulary(words, min_freq=2, max_words=200, min_length=MIN_TOKEN_LENGTH):
    freq = Counter(words)
    selected = [
        word
        for word, count in freq.most_common()
        if count >= min_freq and len(word) >= min_length
    ]
    return selected[:max_words]


def get_word_embeddings(words, language, min_freq=2, max_words=80, min_length=MIN_TOKEN_LENGTH):
    selected_words = _select_vocabulary(words, min_freq, max_words, min_length)
    if not selected_words:
        return selected_words, []

    model = _get_embedding_model(language)
    embeddings = model.encode(selected_words)
    return selected_words, embeddings


def normalize_weights_minmax(G):

    if G.number_of_edges() == 0:
        return G

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    min_w = min(weights)
    max_w = max(weights)

    if max_w == min_w:
        # Все веса одинаковые — ставим 1.0
        for u, v, d in G.edges(data=True):
            d['weight'] = 1.0
        return G

    for u, v, d in G.edges(data=True):
        d['weight'] = (d['weight'] - min_w) / (max_w - min_w)

    return G

def build_semantic_graph(
        words,
        language,
        top_k: int = 4,
        min_similarity: float = None,
        auto_threshold_percentile: float = 85.0,
        min_freq: int = 2,
        max_words: int = 200,
        min_length: int = MIN_TOKEN_LENGTH,
        normalize_weights: bool = True,
        verbose: bool = True,
):
    unique_words = _select_vocabulary(words, min_freq=min_freq, max_words=max_words, min_length=min_length)

    G = nx.Graph()
    for word in unique_words:
        G.add_node(word)

    if len(unique_words) < 2:
        if verbose:
            print("Not enough unique words to build semantic graph.")
        return G

    model = _get_embedding_model(language)
    embeddings = model.encode(unique_words)

    # АВТОМАТИЧЕСКИЙ ПОДБОР ПОРОГА
    if min_similarity is None:
        min_similarity = get_similarity_threshold(
            embeddings=embeddings,
            percentile=auto_threshold_percentile,
        )
        if verbose:
            print(f"\n--- Auto threshold selection ---")
            print(f"Strategy: percentile (p={auto_threshold_percentile}%)")
            print(f"Selected threshold: {min_similarity:.4f}")

            stats = get_similarity_stats(embeddings)
            print(f"Similarity stats: min={stats['min']:.3f}, mean={stats['mean']:.3f}, "
                  f"median={stats['median']:.3f}, max={stats['max']:.3f}")
            print(f"Pairs analyzed: {stats['num_pairs']}")

    sim_matrix = cosine_similarity(embeddings)

    edges_added = 0
    for i, word in enumerate(unique_words):
        neighbors = sorted(
            enumerate(sim_matrix[i]),
            key=lambda item: item[1],
            reverse=True,
        )

        for j, sim in neighbors[1:top_k + 1]:
            if sim >= min_similarity:
                G.add_edge(word, unique_words[j], weight=float(sim))
                edges_added += 1

    isolates = list(nx.isolates(G))
    if isolates and verbose:
        print(f"Removed {len(isolates)} isolated nodes (no edges above threshold)")
    G.remove_nodes_from(isolates)

    if verbose:
        print(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print("-----------------------------------\n")

    if normalize_weights and G.number_of_edges() > 0:
        G = normalize_weights_minmax(G)
        if verbose:
            print(f"Weights normalized to [0, 1] range")

    return G