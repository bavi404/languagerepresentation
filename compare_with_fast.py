import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load FastText Word Embeddings
def load_fasttext_embeddings(fasttext_file, vocab_limit=50000):
    """Loads FastText embeddings into a dictionary."""
    embeddings = {}
    with open(fasttext_file, "r", encoding="utf-8") as f:
        next(f)  # Skip first line (metadata)
        for i, line in enumerate(f):
            values = line.strip().split()
            word = values[0].lower()  # Convert words to lowercase
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
            if len(embeddings) >= vocab_limit:
                break  # Limit vocabulary size for efficiency
    print(f"Loaded {len(embeddings)} words from {fasttext_file}. Example words: {list(embeddings.keys())[:10]}")
    return embeddings

# Compute Cosine Similarity Between Two Words
def compute_cosine_similarity(word1, word2, embeddings):
    """Computes cosine similarity between two words for FastText."""
    word1, word2 = word1.lower(), word2.lower()
    if word1 not in embeddings or word2 not in embeddings:
        return None  # Skip words not in embeddings
    vec1 = embeddings[word1].reshape(1, -1)
    vec2 = embeddings[word2].reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

# Evaluate FastText Embeddings Against Human Judgments
def evaluate_fasttext_quality(dataset_path, embeddings):
    """Evaluates FastText embeddings using human-labeled datasets."""
    
    # Automatically detect delimiter
    delimiter = "\t" if dataset_path.endswith((".txt", ".tab")) else ","
    
    # Load dataset
    df = pd.read_csv(dataset_path, delimiter=delimiter)

    # Normalize column names to avoid mismatches
    df.columns = df.columns.str.lower().str.strip()
    
    # Identify dataset type dynamically
    possible_word_cols = [col for col in df.columns if "word" in col or "term" in col]
    possible_score_cols = [col for col in df.columns if "sim" in col or "score" in col]

    if len(possible_word_cols) < 2 or len(possible_score_cols) < 1:
        print(f"Unknown dataset format for {dataset_path}")
        print("Available columns:", df.columns.tolist())
        return
    
    word1_col, word2_col = possible_word_cols[:2]
    score_col = possible_score_cols[0]

    # Convert words to lowercase
    df[word1_col] = df[word1_col].str.lower()
    df[word2_col] = df[word2_col].str.lower()

    # Remove rows where similarity score is not a number
    df = df[[word1_col, word2_col, score_col]].copy()
    df = df[pd.to_numeric(df[score_col], errors='coerce').notnull()]
    df[score_col] = df[score_col].astype(float)

    predicted_similarities, human_scores = [], []
    missing_pairs = 0

    for _, row in df.iterrows():
        word1, word2, human_score = row[word1_col], row[word2_col], row[score_col]
        cos_sim = compute_cosine_similarity(word1, word2, embeddings)

        if cos_sim is not None:
            predicted_similarities.append(float(cos_sim))
            human_scores.append(float(human_score))
        else:
            missing_pairs += 1

    if len(predicted_similarities) == 0 or len(human_scores) == 0:
        print(f"No valid word pairs found in {dataset_path}. Check dataset format!")
        return

    predicted_similarities = np.array(predicted_similarities, dtype=np.float32)
    human_scores = np.array(human_scores, dtype=np.float32)

    correlation = np.corrcoef(predicted_similarities, human_scores)[0, 1]
    print(f"Correlation with human similarity scores ({dataset_path} - FastText): {correlation:.4f}")
    print(f"Skipped {missing_pairs} word pairs not found in FastText vocabulary.\n")

if __name__ == "__main__":
    print("Loading FastText embeddings...")

    # Load FastText embeddings (Make sure the path is correct)
    fasttext_embeddings = load_fasttext_embeddings("cc.en.300.vec")

    print("\nChecking first 10 words in SimLex-999:")
    print(pd.read_csv("SimLex-999.txt", delimiter="\t").head(10))

    print("\nChecking first 10 words in WordSim-353:")
    print(pd.read_csv("wordsim353/combined.tab", delimiter="\t").head(10))

    print("\nEvaluating FastText using SimLex-999...")
    evaluate_fasttext_quality("SimLex-999.txt", fasttext_embeddings)

    print("\nEvaluating FastText using WordSimilarity-353...")
    evaluate_fasttext_quality("wordsim353/combined.tab", fasttext_embeddings)

    print("Evaluation Completed Successfully!")
