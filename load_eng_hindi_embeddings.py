import numpy as np

def load_embeddings(file_path, vocab_limit=50000):
    """Loads pre-trained word embeddings into a dictionary."""
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header (some FastText files have a header)
        for i, line in enumerate(f):
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
            if len(embeddings) >= vocab_limit:
                break  # Limit vocabulary size for efficiency
    return embeddings

# ✅ Load English and Hindi embeddings
eng_embeddings = load_embeddings("cc.en.300.vec")
hin_embeddings = load_embeddings("cc.hi.300.vec")

print(f"Loaded {len(eng_embeddings)} English embeddings.")
print(f"Loaded {len(hin_embeddings)} Hindi embeddings.")
