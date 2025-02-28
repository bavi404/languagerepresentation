import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_embeddings(embedding_file, vocab_file):
    embeddings = np.load(embedding_file)
    vocab = np.load(vocab_file, allow_pickle=True).item()
    return embeddings, vocab

def visualize_embeddings(embeddings, vocab, method="pca"):
    """Visualizes word embeddings using PCA or t-SNE."""
    words = ["king", "queen", "man", "woman", "apple", "banana", "computer", "laptop", "car", "road", "doctor", "nurse"]
    word_indices = [vocab[word] for word in words if word in vocab]

    selected_embeddings = embeddings[word_indices]
    
    if method == "pca":
        reducer = PCA(n_components=2)
        title = "PCA Visualization of Word Embeddings"
    else:
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
        title = "t-SNE Visualization of Word Embeddings"
    
    reduced_embeddings = reducer.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    embeddings, vocab = load_embeddings("word_embeddings_win5.npy", "vocab_win5.npy")
    
    print("PCA Visualization:")
    visualize_embeddings(embeddings, vocab, method="pca")

    print("t-SNE Visualization:")
    visualize_embeddings(embeddings, vocab, method="tsne")
