# ML Clustering Vocabulary

Проект по теме: "Кластеризация лексики на основе графов и сообществ".

Идея проекта:
- текст проходит предобработку;
- из слов строится граф;
- на графе ищутся сообщества;
- найденные кластеры интерпретируются как тематические группы или смысловые поля.

## Что умеет проект

- Автоматически определяет язык входного текста: русский или английский.
- Для английского текста использует текущую модель эмбеддингов `all-MiniLM-L6-v2`.
- Для русского текста использует multilingual-модель `paraphrase-multilingual-MiniLM-L12-v2`.
- Поддерживает два режима графа:
  - `baseline`: граф совместной встречаемости слов в окне;
  - `semantic`: граф семантической близости по эмбеддингам.
- Поддерживает два алгоритма поиска сообществ:
  - Louvain;
  - Girvan-Newman.
- Сохраняет визуализации в папку `outputs/`, поэтому проект запускается и без GUI.

## Структура

- [main.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/main.py) — основной сценарий эксперимента.
- [config.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/config.py) — параметры проекта.
- [preprocessing/text_processor.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/preprocessing/text_processor.py) — определение языка и предобработка.
- [graph/graph_builder.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/graph/graph_builder.py) — baseline-граф по окну совместной встречаемости.
- [graph/semantic_graph_builder.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/graph/semantic_graph_builder.py) — semantic-граф по эмбеддингам.
- [clustering/community_detection.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/clustering/community_detection.py) — поиск сообществ.
- [analysis/cluster_analysis.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/analysis/cluster_analysis.py) — топ-слова и метрики кластеров.
- [visualization/visualizer.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/visualization/visualizer.py) — визуализация графа.
- [visualization/embedding_visualizer.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/visualization/embedding_visualizer.py) — PCA-проекция эмбеддингов.

## Установка

1. Установить зависимости:

```powershell
pip install -r requirements_clean.txt
pip install -r requirements.txt
```

2. Для английского языка нужна модель spaCy:

```powershell
python -m spacy download en_core_web_sm
```

3. Для русского языка желательно установить модель spaCy:

```powershell
python -m spacy download ru_core_news_sm
```

Если русская spaCy-модель не установлена, проект все равно запустится: будет использован `spacy.blank("ru")` без POS-разметки. Это менее точно, но устойчиво для демонстрации.

## Запуск

```powershell
python main.py
```

Входной текст по умолчанию читается из [data/sample.txt](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/data/sample.txt).

## Основные параметры

Параметры находятся в [config.py](/D:/PROJEKTS%20PYTHON/ML_Clusteringv_Vocabulary/config.py):

- `LANGUAGE_MODE = "auto"` — автоопределение языка.
- `GRAPH_MODE = "semantic"` — режим графа.
- `MIN_WORD_FREQ` — минимальная частота слова.
- `MAX_UNIQUE_WORDS` — максимальный размер словаря.
- `SEMANTIC_TOP_K` — число ближайших соседей в semantic-графе.
- `SEMANTIC_MIN_SIM` — порог cosine similarity.
- `SAVE_PLOTS` — сохранять изображения в `outputs/`.
- `SHOW_PLOTS` — показывать графики в GUI.

## Что выводится

Во время запуска проект печатает:

- обнаруженный язык текста;
- доступность POS-теггера;
- число слов после предобработки;
- статистику графа;
- кластеры и их ключевые слова;
- число кластеров, средний размер и modularity.

Дополнительно в `outputs/` сохраняются:

- PCA-визуализация эмбеддингов;
- граф Louvain;
- граф Girvan-Newman.

## Методические замечания

- `baseline` подходит как простая контрольная модель.
- `semantic` лучше соответствует теме про смысловые поля.
- Для русских текстов качество заметно выше при наличии `ru_core_news_sm`.
- Для защиты полезно сравнивать `baseline` и `semantic` на одном и том же корпусе.
