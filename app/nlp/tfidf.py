import numpy as np
import math

def tf_idf(texts):
    """
    Вычисление TF-IDF для списка текстов
    Возвращает словарь с результатами
    """
    # Токенизация
    docs = [text.lower().split() for text in texts]
    
    # Построение словаря
    vocab = []
    for doc in docs:
        for word in doc:             
            if word not in vocab:
                vocab.append(word)
    
    N = len(docs)
    V = len(vocab)
    
    # TF матрица
    tf_matrix = np.zeros((N, V))
    for i, doc in enumerate(docs):
        words = doc
        words_count = len(words)
        if words_count == 0:
            continue
        # Используем set для оптимизации
        for word in set(words):
            j = vocab.index(word)
            tf_value = words.count(word) / words_count
            tf_matrix[i, j] = tf_value
    
    # DF вектор
    df_vector = np.zeros(V)
    for j, word in enumerate(vocab):
        count = 0
        for doc in docs:
            if word in doc:            
                count += 1
        df_vector[j] = count
    
    # IDF вектор
    idf_vector = np.zeros(V)
    for j in range(V):
        n = df_vector[j]
        idf_vector[j] = math.log10(N / n) if n > 0 else 0
    
    # TF-IDF матрица
    tfidf_matrix = np.zeros((N, V))
    for i in range(N):
        for j in range(V):
            tfidf_matrix[i, j] = tf_matrix[i, j] * idf_vector[j]
    
    # Преобразуем numpy массивы в списки для JSON сериализации
    return {
        "vocabulary": vocab,
        "tf_matrix": tf_matrix.tolist(),
        "idf_vector": idf_vector.tolist(),
        "tfidf_matrix": tfidf_matrix.tolist(),
        "num_documents": N,
        "vocabulary_size": V
    }