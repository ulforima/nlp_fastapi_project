import numpy as np

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

