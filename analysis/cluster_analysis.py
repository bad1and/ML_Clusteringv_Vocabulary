# получить самые важные слова в кластерах
def get_top_words(G, clusters, top_n=5):
    result = {} # итоговое множество

    # для каждого кластера
    for cid, words in clusters.items():

        # сортировка важности слов
        scored = sorted(
            words,
            key=lambda w: G.degree(w, weight="weight"),
            reverse=True
        )
        result[cid] = scored[:top_n] # выбираем первые N важнейших слов

    return result