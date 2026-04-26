import matplotlib.pyplot as plt
import networkx as nx

from analysis.cluster_analysis import get_top_words
from clustering.community_detection import detect_communities
from config import (
    COOCCURRENCE_MIN_WEIGHT,
    GRAPH_MODE,
    MAX_UNIQUE_WORDS,
    MIN_WORD_FREQ,
    SEMANTIC_MIN_SIM,
    SEMANTIC_TOP_K,
    TOP_N_WORDS,
)
from graph.graph_builder import build_graph, filter_graph
from graph.semantic_graph_builder import build_semantic_graph
from preprocessing.text_processor import preprocess_text
from visualization.visualizer import draw_graph


def print_graph_stats(G):
    print("\n--- Graph stats ---")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    if G.number_of_nodes() > 0:
        print("Connected components:", nx.number_connected_components(G))
        largest = max(nx.connected_components(G), key=len)
        print("Largest component size:", len(largest))

    if G.number_of_edges() > 0:
        weights = [data["weight"] for _, _, data in G.edges(data=True)]
        print("Min weight:", round(min(weights), 3))
        print("Max weight:", round(max(weights), 3))
        print("Avg weight:", round(sum(weights) / len(weights), 3))


def filter_weak_edges(G, threshold=0.4):
    edges_to_remove = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("weight", 0) < threshold
    ]
    G.remove_edges_from(edges_to_remove)
    return G

def run_clustering(G, method):
    print(f"\n--- {method} ---")
    clusters, partition = detect_communities(G, method=method)
    top_words = get_top_words(G, clusters, TOP_N_WORDS)

    for cid, words in top_words.items():
        print(f"Cluster {cid}: {words}")

    print(f"Total clusters: {len(clusters)}")
    draw_graph(G, partition, title=method)


def main():
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    words = preprocess_text(text, debug_pos=False)
    print("Words after preprocessing:", len(words))

    if GRAPH_MODE == "baseline":
        # Тупая модель для презентации: связывает слова, которые стоят рядом.
        G = build_graph(words)
        G = filter_graph(G, min_weight=COOCCURRENCE_MIN_WEIGHT)

    elif GRAPH_MODE == "semantic":
        # Умная модель: связывает слова по смысловой близости embedding'ов.
        # Старый filter_graph(min_weight=2) здесь НЕ используем, потому что
        # cosine similarity обычно меньше 1.
        G = build_semantic_graph(
            words,
            top_k=SEMANTIC_TOP_K,
            min_similarity=SEMANTIC_MIN_SIM,
            min_freq=MIN_WORD_FREQ,
            max_words=MAX_UNIQUE_WORDS,
        )

        G = filter_weak_edges(G, threshold=0.50)

    else:
        raise ValueError("GRAPH_MODE must be 'baseline' or 'semantic'")

    print_graph_stats(G)

    if G.number_of_edges() == 0:
        print("\nGraph has no edges. Try lowering SEMANTIC_MIN_SIM or increasing SEMANTIC_TOP_K.")
        return

    run_clustering(G, method="louvain")

    # Girvan-Newman сильно медленнее и хуже масштабируется. Для защиты можно
    # оставить как сравнение, но если тормозит — закомментировать.
    if G.number_of_nodes() <= 120:
        run_clustering(G, method="girvan_newman")
    else:
        print("\nGirvan-Newman skipped: graph is too large.")


if __name__ == "__main__":
    main()
    plt.show()
