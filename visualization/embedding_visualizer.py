import matplotlib.pyplot as plt # для графиков
from sklearn.decomposition import PCA # метод понижения размерности эмбеддингов

# визуализация эмбеддингов
def visualize_embeddings(words, embeddings, title="Sentence-BERT embeddings"):

    # если мало слов
    if len(words) < 2:
        print("Not enough words to visualize embeddings.")
        return

    pca = PCA(n_components=2, random_state=42) # сжатие с фиксацией случайности
    points = pca.fit_transform(embeddings) # проецирование на 2 главные компоненты

    plt.figure(figsize=(14, 9)) # создание окна

    # проходим по словам
    for i, word in enumerate(words):
        x, y = points[i] # берем координаты точек
        plt.scatter(x, y) # рисуем точки
        plt.text(x + 0.01, y + 0.01, word, fontsize=8) # подписываем точки

    plt.title(title) # заголовок графика
    plt.xlabel("PCA component 1") # 1я главная компонента
    plt.ylabel("PCA component 2") # 2я главная компонента
    plt.grid(True) # координатная сетка
    plt.show(block=False) # не блокирует работу программы
    plt.pause(0.1) # время отрисовки окна