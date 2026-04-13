def get_top_words(G, clusters, top_n=5):
    result = {}

    for cid, words in clusters.items():
        scored = sorted(
            words,
            key=lambda w: G.degree(w, weight="weight"),
            reverse=True
        )
        result[cid] = scored[:top_n]

    return result