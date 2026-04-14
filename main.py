from analysis.cluster_analysis import get_top_words
from clustering.community_detection import detect_communities
from config import TOP_N_WORDS
from graph.graph_builder import build_graph, filter_graph
from preprocessing.text_processor import preprocess_text
from visualization.visualizer import draw_graph
import matplotlib.pyplot as plt


def main():
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    from collections import Counter

    def filter_rare_words(words, min_freq=2):
        freq = Counter(words)
        return [w for w in words if freq[w] >= min_freq]

    # 1. Предобработка
    words = preprocess_text(text, debug_pos=True)
    words = filter_rare_words(words, min_freq=2)

    # 2. Граф
    G = build_graph(words)
    G = filter_graph(G, min_weight=2)


    # 3. Louvain
    print("\n--- Louvain ---")
    clusters_l, partition_l = detect_communities(G, method="louvain")

    top_l = get_top_words(G, clusters_l, TOP_N_WORDS)

    for cid, words in top_l.items():
        print(f"Cluster {cid}: {words}")

    print(f"Total clusters: {len(clusters_l)}")

    draw_graph(G, partition_l, title="Louvain")

    # 4. Girvan-Newman
    print("\n--- Girvan-Newman ---")
    clusters_g, partition_g = detect_communities(G, method="girvan_newman")

    top_g = get_top_words(G, clusters_g, TOP_N_WORDS)

    for cid, words in top_g.items():
        print(f"Cluster {cid}: {words}")

    print(f"Total clusters: {len(clusters_g)}")

    draw_graph(G, partition_g, title="Girvan-Newman")


if __name__ == "__main__":
    main()
    plt.show()
