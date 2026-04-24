from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_semantic_graph(words, top_k=5):
    G = nx.Graph()

    unique_words = list(set(words))
    embeddings = model.encode(unique_words)

    sim_matrix = cosine_similarity(embeddings)

    for i in range(len(unique_words)):
        sims = list(enumerate(sim_matrix[i]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)

        # берем топ ближайших слов
        for j, sim in sims[1:top_k+1]:
            w1 = unique_words[i]
            w2 = unique_words[j]

            G.add_edge(w1, w2, weight=float(sim))

    return G