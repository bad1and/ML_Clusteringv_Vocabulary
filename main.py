import matplotlib.pyplot as plt # библиотека для графиков
import networkx as nx # библиотека для графов

from analysis.cluster_analysis import get_top_words # для кластеров
from clustering.community_detection import detect_communities # для алгоритмов поиска сообществ

# импорт настроек из config
from config import (
    COOCCURRENCE_MIN_WEIGHT,
    GRAPH_MODE,
    MAX_UNIQUE_WORDS,
    MIN_WORD_FREQ,
    SEMANTIC_MIN_SIM,
    SEMANTIC_NEIGHBORS,
    TOP_N_WORDS,
)

from graph.graph_builder import build_graph, filter_graph # для тупой модели
from graph.semantic_graph_builder import build_semantic_graph, get_word_embeddings # для умной модели
from preprocessing.text_processor import preprocess_text # для предобработки текста
from visualization.visualizer import draw_graph # для виртуализации графа
from visualization.embedding_visualizer import visualize_embeddings # для виртуализации слов

# информация о графе
def print_graph_stats(G):

    # вершины и ребра
    print("\n--- Graph stats ---")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # кластеры и размер наибольшего кластера
    if G.number_of_nodes() > 0:
        print("Connected components:", nx.number_connected_components(G))
        largest = max(nx.connected_components(G), key=len)
        print("Largest component size:", len(largest))

    # статистика весов
    if G.number_of_edges() > 0:
        weights = [data["weight"] for _, _, data in G.edges(data=True)]
        print("Min weight:", round(min(weights), 3))
        print("Max weight:", round(max(weights), 3))
        print("Avg weight:", round(sum(weights) / len(weights), 3))

# удаление лишних ребер (по косинусному сходству)
def filter_weak_edges(G, threshold=0.4):

    # формируем массив ребер со слабой связью
    edges_to_remove = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("weight", 0) < threshold
    ]

    # удаляем массив ребер со слабой связью
    G.remove_edges_from(edges_to_remove)
    return G

# алгоритм поиска сообществ
def run_clustering(G, method):
    print(f"\n--- {method} ---")

    # получаем кластеры слов
    clusters, partition = detect_communities(G, method=method)

    # получаем и выводим N важнейших слов для каждого кластера
    top_words = get_top_words(G, clusters, TOP_N_WORDS)
    for cid, words in top_words.items():
        print(f"Cluster {cid}: {words}")

    print(f"Total clusters: {len(clusters)}") # выводим общее кол-во кластеров
    draw_graph(G, partition, title=method) # рисуем граф

# основная программа
def main():

    # считываем файл с текстом
    with open("data/sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # предобработка текста
    words = preprocess_text(text, debug_pos=False)
    print("Words after preprocessing:", len(words))


    # визуализация эмбеддингов
    selected_words, embeddings = get_word_embeddings(
        words,
        min_freq=MIN_WORD_FREQ,
        max_words=MAX_UNIQUE_WORDS,
    )
    visualize_embeddings(selected_words, embeddings)

    # Тупая модель: связывает слова, которые стоят рядом.
    if GRAPH_MODE == "baseline":
        G = build_graph(words)
        G = filter_graph(G, min_weight=COOCCURRENCE_MIN_WEIGHT) # удаляем лишние ребра

    # Умная модель: связывает слова по смысловой близости эмбеддингов.
    elif GRAPH_MODE == "semantic":
        G = build_semantic_graph(
            words,
            top_k=SEMANTIC_NEIGHBORS,
            min_similarity=SEMANTIC_MIN_SIM,
            min_freq=MIN_WORD_FREQ,
            max_words=MAX_UNIQUE_WORDS,
        )
        G = filter_weak_edges(G, threshold=0.50) # удаляем лишние ребра

    # при ошибке
    else:
        raise ValueError("GRAPH_MODE must be 'baseline' or 'semantic'")

    print_graph_stats(G) # информация о графе

    # если граф без ребер
    if G.number_of_edges() == 0:
        print("\nGraph has no edges. Try lowering SEMANTIC_MIN_SIM or increasing SEMANTIC_NEIGHBORS.")
        return

    # кластеризация методом Лувена
    run_clustering(G, method="louvain")

    # кластеризация методом Гирвана-Ньюмана (если немного связей)
    if G.number_of_nodes() <= 120:
        run_clustering(G, method="girvan_newman")
    else:
        print("\nGirvan-Newman skipped: graph is too large.")


# удерживать окна открытыми до ручного закрытия
if __name__ == "__main__":
    main()
    plt.show()
