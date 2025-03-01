import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
def load_fasttext_embeddings(glove_file):
    embeddings = {}
    with open(fasttext_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0].lower()
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_fasttext_embeddings("cc.en.300.vec")

# Define gendered word sets
male_words = ["man", "he", "him", "father", "boy", "brother"]
female_words = ["woman", "she", "her", "mother", "girl", "sister"]

# Define profession words
professions = ["doctor", "nurse", "engineer", "teacher", "CEO", "homemaker"]

# Compute Bias Score for each profession
bias_scores = {}
for profession in professions:
    if profession in fasttext_embeddings:
        prof_vec = fasttext_embeddings[profession].reshape(1, -1)
        male_sim = np.mean([cosine_similarity(prof_vec, fasttext_embeddings[m].reshape(1, -1))[0][0] for m in male_words if m in fasttext_embeddings])
        female_sim = np.mean([cosine_similarity(prof_vec, fasttext_embeddings[f].reshape(1, -1))[0][0] for f in female_words if f in fasttext_embeddings])
        bias_scores[profession] = male_sim - female_sim

# Print results
for profession, score in bias_scores.items():
    print(f"Bias Score for {profession}: {score:.4f}")
