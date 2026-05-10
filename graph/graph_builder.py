import networkx as nx # библиотека для графов
from config import COOCCURRENCE_NEIGHBORS # импорт размера окна

# построение графа
def build_graph(words):
    G = nx.Graph()

    # для каждого слова
    for i in range(len(words)):

        # берем его ближайших N соседей
        for j in range(i + 1, i + COOCCURRENCE_NEIGHBORS):

            # если не вышли за предел кол-ва слов
            if j < len(words):
                w1, w2 = words[i], words[j] # две вершины для связи

                distance = j - i
                weight = 1 / distance  # усиление близких слов

                # если уже есть связь - усиливаем вес
                if G.has_edge(w1, w2):
                    G[w1][w2]["weight"] += weight

                # если нет связи - создаем
                else:
                    G.add_edge(w1, w2, weight=weight)

    return G

# удаление лишних ребер
def filter_graph(G, min_weight=2):

    # формируем массив ребер со слабой связью
    edges_to_remove = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["weight"] < min_weight
    ]

    # удаляем массив ребер со слабой связью
    G.remove_edges_from(edges_to_remove)
    return G