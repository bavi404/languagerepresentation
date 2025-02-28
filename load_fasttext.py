import numpy as np

def load_fasttext_embeddings(fasttext_file, vocab_limit=50000):
    """Loads FastText embeddings into a dictionary."""
    embeddings = {}
    with open(fasttext_file, "r", encoding="utf-8") as f:
        next(f)  # Skip first line (contains metadata: total words & dimensions)
        for i, line in enumerate(f):
            values = line.strip().split()
            word = values[0].lower()  # Convert words to lowercase
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
            if len(embeddings) >= vocab_limit:
                break  # Limit vocabulary size for efficiency
    return embeddings

if __name__ == "__main__":
    fasttext_embeddings = load_fasttext_embeddings("cc.en.300.vec")  # Change to correct file path
    print(f"Loaded {len(fasttext_embeddings)} words from FastText.")
