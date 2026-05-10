from collections import Counter # словарь для подсчета частот слов

import networkx as nx
from sentence_transformers import SentenceTransformer # библиотека для sentence-BERT
from sklearn.metrics.pairwise import cosine_similarity # для подсчета косинусного сходства

# English model. Для русского текста лучше заменить на:
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer("all-MiniLM-L6-v2")

# формируем итоговый массив слов
def _select_vocabulary(words, min_freq=2, max_words=200):
    freq = Counter(words) # подсчет кол-ва слов

    # итоговый массив (отсекаем короткие и редкие слова)
    selected = [
        word
        for word, count in freq.most_common()
        if count >= min_freq and len(word) > 2
    ]
    return selected[:max_words] # отсекаем первые N самых часто-встречаемых слов

# для
def get_word_embeddings(words, min_freq=2, max_words=80):
    selected_words = _select_vocabulary(words, min_freq, max_words) # получаем итоговый массив слов
    embeddings = model.encode(selected_words) # получает смысловой вектор слов
    return selected_words, embeddings

# построение графа
def build_semantic_graph(words, top_k=4, min_similarity=0.20, min_freq=2, max_words=200):

    selected_words = _select_vocabulary(words, min_freq=min_freq, max_words=max_words) # получаем итоговый массив слов

    # формирование вершин графа
    G = nx.Graph()
    for word in selected_words:
        G.add_node(word)

    # если кол-во слов влишком мало
    if len(selected_words) < 2:
        return G

    embeddings = model.encode(selected_words) # получает смысловой вектор слов
    sim_matrix = cosine_similarity(embeddings) # получает матрицу с косинусным сходством слов

    # формирование ребер графа
    for i, word in enumerate(selected_words):

        # формирование соседей слов
        neighbors = sorted(
            enumerate(sim_matrix[i]), # сортировка по убыванию схожести
            key=lambda item: item[1],
            reverse=True,
        )

        # берем самых близких соседей
        for j, sim in neighbors[1 : top_k + 1]:
            if sim >= min_similarity:
                G.add_edge(word, selected_words[j], weight=float(sim))

    # удаление изолированных вершин
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    return G
