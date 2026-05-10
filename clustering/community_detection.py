from collections import defaultdict # словарь с автоматической генерацией ключей

import community as community_louvain # для алгоритма Лувена
import networkx as nx # библиотека для графов

# форматирование вывода кластеров
def _clusters_from_partition(partition):
    clusters = defaultdict(list) #
    for word, cluster_id in partition.items():
        clusters[cluster_id].append(word)
    return clusters

# метод Лувена
def louvain_method(G):

    # если пустой граф
    if G.number_of_nodes() == 0:
        return defaultdict(list), {}

    # кластеризация с фиксацией случайности
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    clusters = _clusters_from_partition(partition) # форматирование вывода
    return clusters, partition

# метод Гирвана-Ньюмана
def girvan_newman_method(G, level=1):

    # если пустой граф
    if G.number_of_nodes() == 0:
        return defaultdict(list), {}

    # если нет ребер
    if G.number_of_edges() == 0:

        # каждое слово - отдельный кластер
        partition = {node: i for i, node in enumerate(G.nodes())}
        return _clusters_from_partition(partition), partition

    # разбиение до N кластеров через удаление мостов
    comp = nx.community.girvan_newman(G)
    communities = None
    for _ in range(level):
        communities = next(comp)

    # кластеризация
    clusters = defaultdict(list)
    partition = {}

    # для каждого сообщества
    for i, community in enumerate(communities):

        # преобразуем множества в словари
        for node in community:
            clusters[i].append(node)
            partition[node] = i

    return clusters, partition

# определение выбранного метода
def detect_communities(G, method="louvain"):
    if method == "louvain":
        return louvain_method(G)
    if method == "girvan_newman":
        return girvan_newman_method(G)
    raise ValueError("Unknown method")
