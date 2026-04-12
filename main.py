from preprocessing.text_processor import preprocess_text
from graph.graph_builder import build_graph
from clustering.community_detection import detect_communities
from visualization.visualizer import draw_graph

def main():
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    words = preprocess_text(text)
    G = build_graph(words)
    clusters, partition = detect_communities(G)

    print("Кластеры:")
    for cid, words in clusters.items():
        print(cid, words)

    draw_graph(G, partition)

if __name__ == "__main__":
    main()