import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embedding_file, vocab_file):
    """Loads embeddings and vocabulary."""
    embeddings = np.load(embedding_file)
    vocab = np.load(vocab_file, allow_pickle=True).item()
    return embeddings, vocab

def compute_similarity(word1, word2, embeddings, vocab):
    """Computes cosine similarity between two words."""
    if word1 not in vocab or word2 not in vocab:
        return None
    vec1 = embeddings[vocab[word1]].reshape(1, -1)
    vec2 = embeddings[vocab[word2]].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

if __name__ == "__main__":
    embeddings, vocab = load_embeddings("word_embeddings_win5.npy", "vocab_win5.npy")
    
    word_pairs = [("king", "queen"), ("apple", "banana"), ("car", "road"), 
                  ("computer", "laptop"), ("doctor", "nurse"), ("dog", "cat")]

    print("\n Cosine Similarity Scores:")
    for word1, word2 in word_pairs:
        sim = compute_similarity(word1, word2, embeddings, vocab)
        print(f"{word1} - {word2}: {sim:.4f}" if sim is not None else f"{word1} or {word2} not found in vocab!")
