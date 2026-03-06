import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns (tfidf_matrix, vocabulary)
    """

    # Handle empty corpus
    if not documents:
        return np.array([]), []

    # Step 1: Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Step 2: Build vocabulary
    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    vocab_index = {word: i for i, word in enumerate(vocab)}

    n_docs = len(documents)
    n_vocab = len(vocab)

    # Step 3: Compute Document Frequency (df)
    df = Counter()
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] += 1

    # Step 4: Compute IDF
    idf = {}
    for word in vocab:
        if df[word] == 0:
            idf[word] = 0
        else:
            idf[word] = math.log(n_docs / df[word])

    # Step 5: Initialize TF-IDF matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab))

    # Step 6: Fill matrix
    for i, doc in enumerate(tokenized_docs):
        if len(doc) == 0:
            continue

        term_counts = Counter(doc)

        for word, count in term_counts.items():
            j = vocab_index[word]

            # TF
            tf = count / len(doc)

            # TF-IDF
            tfidf_matrix[i][j] = tf * idf[word]

    return tfidf_matrix, vocab