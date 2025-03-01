import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(embedding_file, vocab_file):
    """Loads word embeddings and vocabulary."""
    embeddings = np.load(embedding_file)
    vocab = np.load(vocab_file, allow_pickle=True).item()
    return embeddings, vocab

def compute_cosine_similarity(word1, word2, embeddings, vocab):
    """Computes cosine similarity between two words."""
    if word1 not in vocab or word2 not in vocab:
        return None  # Skip words not in vocabulary
    vec1 = embeddings[vocab[word1]].reshape(1, -1)
    vec2 = embeddings[vocab[word2]].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def evaluate_embedding_quality(dataset_path, embeddings, vocab):
    """Evaluates embeddings using human-labeled datasets."""
    
    delimiter = "\t" if dataset_path.endswith(".txt") or dataset_path.endswith(".tab") else ","
    
    df = pd.read_csv(dataset_path, delimiter=delimiter)
    
    df.columns = df.columns.str.lower()
    
    if "simlex-999" in dataset_path.lower():
        word1_col, word2_col, score_col = "word1", "word2", "simlex999"
    elif "wordsim353" in dataset_path.lower():
        word1_col, word2_col, score_col = "word 1", "word 2", "human (mean)"
    else:
        print(f"Error: Unknown dataset format for {dataset_path}")
        return
    
    predicted_similarities, human_scores = [], []
    missing_pairs = 0

    for _, row in df.iterrows():
        word1, word2, human_score = row[word1_col], row[word2_col], row[score_col]
        cos_sim = compute_cosine_similarity(word1, word2, embeddings, vocab)
        
        if cos_sim is not None:
            predicted_similarities.append(cos_sim)
            human_scores.append(human_score)
        else:
            missing_pairs += 1  

    correlation = np.corrcoef(predicted_similarities, human_scores)[0, 1]
    print(f"Correlation with human similarity scores ({dataset_path}): {correlation:.4f}")
    print(f"Skipped {missing_pairs} word pairs not found in vocabulary.\n")

if __name__ == "__main__":
    embeddings, vocab = load_embeddings("word_embeddings_win5.npy", "vocab_win5.npy")
    
    simlex_path = "SimLex-999.txt"
    wordsim_path = "wordsim353/combined.tab"  # Corrected file path for WordSimilarity-353

    print("Evaluating using SimLex-999...")
    evaluate_embedding_quality(simlex_path, embeddings, vocab)
    
    print("Evaluating using WordSimilarity-353...")
    evaluate_embedding_quality(wordsim_path, embeddings, vocab)
