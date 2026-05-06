import spacy # для обработки текста
from config import ALLOWED_POS # импорт допустимых частей речи из config

nlp = spacy.load("en_core_web_sm") # загружаем предобученную модель (англ)

# функция предобработки текста
def preprocess_text(text, debug_pos=False):
    doc = nlp(text) # запуск предобработки

    # вывод таблицы частей речи
    if debug_pos:
        print("Token\t\tPOS Tag")
        print("-----------------------")
        for token in doc:
            print(f"{token.text}\t\t{token.pos_}")

    # формируем массив допустимых слов
    words = [
        token.lemma_.lower() # лемматизация слов (с нижним регистром)
        for token in doc # проходим по всем токенам
        if token.is_alpha # если состоит только из букв
        and not token.is_stop # если не стоп-слово
        and token.pos_ in ALLOWED_POS # часть речи допустима
    ]

    return words