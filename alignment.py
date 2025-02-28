import numpy as np
import os
import sys
from scipy.linalg import orthogonal_procrustes

# Ensure UTF-8 Encoding for Windows Terminal Output
sys.stdout.reconfigure(encoding='utf-8')

# Function to Load Word Embeddings
def load_embeddings(file_path, vocab_limit=50000):
    """Loads pre-trained word embeddings from a text file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        exit(1)

    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header if present
        for i, line in enumerate(f):
            values = line.strip().split()
            word = values[0].lower()
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector
            if len(embeddings) >= vocab_limit:
                break

    print(f"Loaded {len(embeddings)} words from {file_path}. Example words: {list(embeddings.keys())[:10]}")
    return embeddings

# Load English & Hindi embeddings
eng_file = r"C:\Users\bavi0\OneDrive\Documents\assignemnt-precog\cc.en.300.vec"
hin_file = r"C:\Users\bavi0\OneDrive\Documents\assignemnt-precog\cc.hi.300.vec"

print("Loading embeddings...")

eng_embeddings = load_embeddings(eng_file)  
hin_embeddings = load_embeddings(hin_file)

print(f"Loaded {len(eng_embeddings)} English embeddings.")
print(f"Loaded {len(hin_embeddings)} Hindi embeddings.")

# Function to Load Bilingual Dictionary
def load_bilingual_dictionary(dict_path, eng_embeddings, hin_embeddings):
    """Loads a bilingual dictionary and extracts aligned word vectors."""
    eng_words, hin_words = [], []
    eng_vectors, hin_vectors = [], []
    missing_words = []

    if not os.path.exists(dict_path):
        print(f"Error: Bilingual dictionary {dict_path} not found!")
        exit(1)

    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            eng_word, hin_word = parts
            eng_word, hin_word = eng_word.lower(), hin_word.lower()

            if eng_word in eng_embeddings and hin_word in hin_embeddings:
                eng_words.append(eng_word)
                hin_words.append(hin_word)
                eng_vectors.append(eng_embeddings[eng_word])
                hin_vectors.append(hin_embeddings[hin_word])
            else:
                missing_words.append((eng_word, hin_word))

    print(f"{len(missing_words)} word pairs were skipped (not in embeddings).")

    print("Examples of missing words:")
    for eng_word, hin_word in missing_words[:10]:
        try:
            print(f"{eng_word} - {hin_word}")
        except UnicodeEncodeError:
            print(f"{eng_word} - (Unicode Error: Cannot display Hindi characters)")

    return np.array(eng_vectors), np.array(hin_vectors), eng_words, hin_words

# Load bilingual dictionary
dict_file = r"C:\Users\bavi0\OneDrive\Documents\assignemnt-precog\en-hi.txt"

X_eng, X_hin, eng_words, hin_words = load_bilingual_dictionary(dict_file, eng_embeddings, hin_embeddings)

print(f"Aligned {len(eng_words)} word pairs for training.")

# Apply Procrustes Analysis
if len(X_eng) > 0 and len(X_hin) > 0:
    print("Applying Procrustes Analysis for Alignment...")
    
    # Compute transformation matrix W
    W, _ = orthogonal_procrustes(X_eng, X_hin)

    # Apply transformation to English embeddings
    X_eng_aligned = X_eng @ W

    print("Alignment Complete! English embeddings are now in the Hindi space.")

    # Save aligned embeddings
    np.save("eng_aligned.npy", X_eng_aligned)
    np.save("hin_vectors.npy", X_hin)

    print("Aligned embeddings saved as 'eng_aligned.npy' and 'hin_vectors.npy'.")

else:
    print("No word pairs were aligned. Check dictionary formatting and embeddings.")
