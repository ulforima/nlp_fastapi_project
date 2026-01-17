import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# один раз скачать
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("wordnet")

def process_text(text):
    tokens = word_tokenize(text)

    stemmer = PorterStemmer()
    stems = [stemmer.stem(t) for t in tokens]

    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]

    pos = pos_tag(tokens)
    ner = ne_chunk(pos)

    return {
        "tokens": tokens,
        "stems": stems,
        "lemmas": lemmas,
        "pos_tags": pos,
        "named_entities": str(ner)
    }
