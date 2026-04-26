from collections import Counter

import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# English model. Для русского текста лучше заменить на:
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")


def _select_vocabulary(words, min_freq=2, max_words=200):
    """Берём не случайный set(words), а самые частотные слова в стабильном порядке."""
    freq = Counter(words)
    selected = [
        word
        for word, count in freq.most_common()
        if count >= min_freq and len(word) > 2
    ]
    return selected[:max_words]


def build_semantic_graph(words, top_k=4, min_similarity=0.20, min_freq=2, max_words=200):
    """
    Строит семантический граф: вершины = слова, ребра = top_k ближайших
    соседей по cosine similarity между Sentence-BERT embedding'ами.

    Важно: веса cosine similarity находятся примерно в диапазоне [-1; 1],
    чаще всего 0..1. Поэтому старый filter_graph(min_weight=2) здесь
    применять нельзя.
    """
    unique_words = _select_vocabulary(words, min_freq=min_freq, max_words=max_words)

    G = nx.Graph()
    for word in unique_words:
        G.add_node(word)

    if len(unique_words) < 2:
        return G

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

    # Изолированные вершины чаще всего только портят картинку и Louvain.
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G
