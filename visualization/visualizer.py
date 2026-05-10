import matplotlib.pyplot as plt # для графиков
import networkx as nx # библиотека для графов

# визуализация графа
def draw_graph(G, partition, title="Graph"):

    # если пустой граф
    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to draw.")
        return

    # визуализация (фиксация случайности, оптимальное расстояние между узлами, кол-во итераций)
    pos = nx.spring_layout(G, seed=42, weight="weight", k=0.8, iterations=80)

    colors = [partition.get(node, 0) for node in G.nodes()] # список цветов для кластеров
    raw_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()] # список весов ребер
    widths = [0.5 + 4 * w for w in raw_weights] # толщина ребер в зависимости от веса

    plt.figure(figsize=(13, 9)) # создание окна
    plt.title(title) # заголовок

    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35) # ребра графа
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=550, alpha=0.9) # вершины графа
    nx.draw_networkx_labels(G, pos, font_size=8) # подписи вершин

    plt.axis("off") # убираем координатную ось
    plt.tight_layout() # автоматические отступы
    plt.show(block=False) # не блокирует работу программы
