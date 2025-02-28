import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load aligned English and Hindi embeddings
eng_aligned = np.load("eng_aligned.npy")
hin_vectors = np.load("hin_vectors.npy")

# Load word lists
eng_words = np.load("eng_words.npy", allow_pickle=True)  # List of English words
hin_words = np.load("hin_words.npy", allow_pickle=True)  # List of Hindi words

# Ensure dimensions match
assert eng_aligned.shape == hin_vectors.shape, "Embedding matrices must have the same shape!"

# Compute similarity matrix
similarity_matrix = cosine_similarity(eng_aligned, hin_vectors)

# Find closest Hindi words
nearest_hindi_indices = np.argmax(similarity_matrix, axis=1)
nearest_hindi_words = [hin_words[i] for i in nearest_hindi_indices]

# Compute accuracy
correct_matches = sum(1 for i in range(len(eng_words)) if nearest_hindi_words[i] == hin_words[i])
accuracy = correct_matches / len(eng_words)

print(f"Word Translation Accuracy: {accuracy * 100:.2f}%")
