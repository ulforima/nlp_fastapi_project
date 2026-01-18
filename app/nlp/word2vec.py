import numpy as np
import math


def tf_idf(texts):
    docs = [text.lower().split() for text in texts]

    vocab = []
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab.append(word)

    N = len(docs)
    V = len(vocab)

    tf_matrix = np.zeros((N, V))
    for i, doc in enumerate(docs):
        if not doc:
            continue
        for word in set(doc):
            j = vocab.index(word)
            tf_matrix[i, j] = doc.count(word) / len(doc)

    df_vector = np.zeros(V)
    for j, word in enumerate(vocab):
        df_vector[j] = sum(1 for doc in docs if word in doc)

    idf_vector = np.zeros(V)
    for j in range(V):
        if df_vector[j] > 0:
            idf_vector[j] = math.log10(N / df_vector[j])

    tfidf_matrix = tf_matrix * idf_vector

    return {
        "vocabulary": vocab,
        "tfidf_matrix": tfidf_matrix.tolist()
    }


def bag_of_words(texts):
    vocab = sorted(set(" ".join(texts).lower().split()))
    matrix = np.zeros((len(texts), len(vocab)))

    for i, text in enumerate(texts):
        for word in text.lower().split():
            matrix[i, vocab.index(word)] += 1

    return {
        "vocabulary": vocab,
        "bow_matrix": matrix.tolist()
    }


def word2vec_analysis(texts, method="tfidf"):
    """
    Сравнение TF-IDF и Bag of Words

    """
    
    if not texts:
        return {"error": "Список текстов пуст"}

    tfidf_top_indices = []
    bow_top_indices = []
    tfidf_first_doc = None
    bow_first_doc = None

    N = len(texts)

    
    tfidf_result = tf_idf(texts)
    bow_result = bag_of_words(texts)

    vocab_tfidf = tfidf_result["vocabulary"]
    vocab_bow = bow_result["vocabulary"]

    tfidf_array = np.array(tfidf_result["tfidf_matrix"])
    bow_array = np.array(bow_result["bow_matrix"])

    # топ-слова
    if vocab_tfidf and tfidf_array.shape[0] > 0:
        tfidf_first_doc = tfidf_array[0]
        nz = np.nonzero(tfidf_first_doc)[0]
        tfidf_top_indices = nz[np.argsort(tfidf_first_doc[nz])[::-1]][:10]

    if vocab_bow and bow_array.shape[0] > 0:
        bow_first_doc = bow_array[0]
        nz = np.nonzero(bow_first_doc)[0]
        bow_top_indices = nz[np.argsort(bow_first_doc[nz])[::-1]][:10]

    # основной результат
    if method == "tfidf":
        vectors = tfidf_array.tolist()
        vocabulary = vocab_tfidf
        top_words = [
            {"word": vocabulary[i], "score": float(tfidf_first_doc[i])}
            for i in tfidf_top_indices
        ] if tfidf_first_doc is not None else []

    elif method == "bow":
        vectors = bow_array.tolist()
        vocabulary = vocab_bow
        top_words = [
            {"word": vocabulary[i], "score": float(bow_first_doc[i])}
            for i in bow_top_indices
        ] if bow_first_doc is not None else []

    else:
        return {"error": "method должен быть 'tfidf' или 'bow'"}
    
    return {
        "main_result": {
            "method": method,
            "num_documents": N,
            "vocabulary_size": len(vocabulary),
            "vocabulary": vocabulary,
            "vectors": vectors,
            "top_words": top_words
        },
        "comparison": {
            "same_vocabulary": sorted(vocab_tfidf) == sorted(vocab_bow),
            "tfidf_vocab_size": len(vocab_tfidf),
            "bow_vocab_size": len(vocab_bow),
            "recommendation": "TF-IDF — для релевантности, BOW — для частот"
        }
    }
