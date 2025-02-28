import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def load_embeddings(embedding_file, vocab_file):
    embeddings = np.load(embedding_file)
    vocab = np.load(vocab_file, allow_pickle=True).item()
    return embeddings, vocab

def cluster_words(embeddings, vocab, num_clusters=10):
    """Clusters words using K-Means and prints representative words for each cluster."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    clusters = defaultdict(list)
    for word, idx in vocab.items():
        cluster_id = kmeans.labels_[idx]
        clusters[cluster_id].append(word)

    # Print top words in each cluster
    for cluster_id, words in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(words[:10])}")

if __name__ == "__main__":
    embeddings, vocab = load_embeddings("word_embeddings_win5.npy", "vocab_win5.npy")
    cluster_words(embeddings, vocab, num_clusters=10)
