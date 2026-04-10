from collections import defaultdict

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import spacy

# 1. Загружаем модель
nlp = spacy.load("en_core_web_sm")

# 2. Текст
text = """
Python is a programming language. Python is used for machine learning and data science.
Coffee is very tasty. Tea and coffee are popular drinks.
Now i am programming on python and ML.
"""

# 3. Предобработка
doc = nlp(text)
words = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

# 4. Строим граф
G = nx.Graph()
window_size = 3

for i in range(len(words)):
    for j in range(i + 1, i + window_size):
        if j < len(words):
            w1, w2 = words[i], words[j]
            if G.has_edge(w1, w2):
                G[w1][w2]["weight"] += 1
            else:
                G.add_edge(w1, w2, weight=1)

# 5. Кластеризация (Louvain)
partition = community_louvain.best_partition(G)

# 6. Группировка
clusters = defaultdict(list)
for word, cluster_id in partition.items():
    clusters[cluster_id].append(word)

# 7. Вывод кластеров
print("Кластеры:")
for cid, words in clusters.items():
    print(f"{cid}: {words}")

# 8. Визуализация
pos = nx.spring_layout(G)
colors = [partition[node] for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=colors)
plt.show()
