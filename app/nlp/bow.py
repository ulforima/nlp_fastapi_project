import numpy as np

def bag_of_words(texts):
    vocab = sorted(set(" ".join(texts).lower().split()))
    matrix = np.zeros((len(texts), len(vocab))) # np.zeros — это функция из библиотеки NumPy, которая создает массив заданного размера, заполненный нулями

    for i, text in enumerate(texts): # превращает обычный список (или любой итерируемый объект) в счётчик. возвращает пары: (индекс, значение)
        for word in text.lower().split():
            matrix[i, vocab.index(word)] += 1

    return {
        "vocabulary": vocab,
        "bow_matrix": matrix.tolist()
    }

