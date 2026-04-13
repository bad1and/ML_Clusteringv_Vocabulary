import spacy
from config import ALLOWED_POS

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text, debug_pos=False):
    doc = nlp(text)

    if debug_pos:
        print("Token\t\tPOS Tag")
        print("-----------------------")
        for token in doc:
            print(f"{token.text}\t\t{token.pos_}")

    words = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.pos_ in ALLOWED_POS
    ]

    return words