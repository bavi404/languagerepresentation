import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_embeddings(embedding_file, vocab_file):
    """Loads word embeddings and vocabulary."""
    embeddings = np.load(embedding_file)
    vocab = np.load(vocab_file, allow_pickle=True).item()
    word_to_index = vocab
    index_to_word = {idx: word for word, idx in vocab.items()}  
    return embeddings, word_to_index, index_to_word

def find_similar_words(word, embeddings, word_to_index, index_to_word, top_n=5):
    """Finds top N similar words using cosine similarity."""
    if word not in word_to_index:
        print(f"Word '{word}' not found in vocabulary!")
        return
    
    word_idx = word_to_index[word]
    word_vec = embeddings[word_idx].reshape(1, -1)
    
    similarities = cosine_similarity(word_vec, embeddings)[0]
    similar_word_indices = similarities.argsort()[-top_n-1:-1][::-1]
    similar_words = [index_to_word[i] for i in similar_word_indices]

    print(f"Words most similar to '{word}': {similar_words}")

if __name__ == "__main__":
    window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # Default to 5
    print(f"Evaluating word similarities for window size {window_size}...")

    embeddings, word_to_index, index_to_word = load_embeddings(
        f"word_embeddings_win{window_size}.npy", 
        f"vocab_win{window_size}.npy"
    )

    test_words = ["king", "queen", "apple", "computer"]
    for word in test_words:
        find_similar_words(word, embeddings, word_to_index, index_to_word)
