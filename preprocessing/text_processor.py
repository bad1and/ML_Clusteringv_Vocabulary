import re
from functools import lru_cache

import spacy

from config import (
    EN_ALLOWED_POS,
    LANGUAGE_MODE,
    MIN_TOKEN_LENGTH,
    RU_ALLOWED_POS,
)


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text):
    if LANGUAGE_MODE in {"en", "ru"}:
        return LANGUAGE_MODE

    cyrillic_count = len(CYRILLIC_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))

    if cyrillic_count > latin_count:
        return "ru"
    return "en"


@lru_cache(maxsize=2)
def _load_nlp(language):
    if language == "en":
        return spacy.load("en_core_web_sm"), True

    if language == "ru":
        try:
            return spacy.load("ru_core_news_sm"), True
        except OSError:
            # Tokenization and stop words still work with a blank pipeline.
            return spacy.blank("ru"), False

    raise ValueError(f"Unsupported language: {language}")


def _normalize_token(token, language):
    lemma = token.lemma_.lower().strip()
    if not lemma or lemma == "-pron-":
        lemma = token.text.lower().strip()
    if language == "ru":
        lemma = lemma.replace("ё", "е")
    return lemma


def preprocess_text(text, debug_pos=False):
    language = detect_language(text)
    nlp, has_pos_tagger = _load_nlp(language)
    doc = nlp(text)

    if debug_pos:
        print("Language:", language)
        print("Token\t\tPOS Tag")
        print("-----------------------")
        for token in doc:
            print(f"{token.text}\t\t{token.pos_ or 'N/A'}")

    allowed_pos = EN_ALLOWED_POS if language == "en" else RU_ALLOWED_POS
    words = []

    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue

        lemma = _normalize_token(token, language)
        if len(lemma) < MIN_TOKEN_LENGTH:
            continue

        if has_pos_tagger and token.pos_ not in allowed_pos:
            continue

        words.append(lemma)

    return words, language, has_pos_tagger
