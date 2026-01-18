import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# Загрузка
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("wordnet")


def process_text(text):
    """
    NLP-анализ текста по:
    - токенизация
    - стемминг
    - лемматизация
    - POS-tagging
    - Named Entity Recognition
    """

    # 1. Токенизация по предложениям
    sentences = sent_tokenize(text)

    # 2. Токенизация по словам
    word_tokens = word_tokenize(text)

    # 3. Стемминг - процесс нахождения основы слова путем отсечения флексий (окончаний и суффиксов) по определенным правилам
    porter = PorterStemmer()
    snowball = SnowballStemmer("english")

    porter_stems = [porter.stem(w) for w in word_tokens]
    snowball_stems = [snowball.stem(w) for w in word_tokens]

    # 4. Лемматизация - приведение слова к его начальной «словарной» форме, называемой леммой
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in word_tokens]

    # 5. Part-of-Speech tagging - процесс определения части речи для каждого слова в тексте (существительное, глагол, прилагательное и т.д.) с учетом контекста
    pos_tags = pos_tag(word_tokens)

    # 6. Named Entity Recognition - процесс автоматического поиска и классификации ключевых сущностей в тексте
    ner_tree = ne_chunk(pos_tags)

    return {
        "sentences": sentences,
        "word_tokens": word_tokens,
        "porter_stems": porter_stems,
        "snowball_stems": snowball_stems,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "named_entities": str(ner_tree)
    }
