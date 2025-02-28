import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
import sys

def build_cooccurrence_matrix(input_file, output_matrix, output_vocab, window_size):
    """Builds a sparse co-occurrence matrix from a cleaned text corpus."""
    
    word_counts = Counter()
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            word_counts.update(words)
    
    vocab = {word: i for i, word in enumerate(word_counts.keys())}
    vocab_size = len(vocab)

    print(f"Vocabulary built! {vocab_size} unique words found.")

    co_occurrence_dict = defaultdict(lambda: defaultdict(int))

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            for idx, word in enumerate(words):
                word_idx = vocab[word]
                start = max(idx - window_size, 0)
                end = min(idx + window_size + 1, len(words))
                
                for neighbor in words[start:end]:
                    if neighbor != word:  # Avoid self-co-occurrence
                        neighbor_idx = vocab[neighbor]
                        co_occurrence_dict[word_idx][neighbor_idx] += 1

    print("Co-occurrence dictionary built! Converting to sparse matrix...")

    rows, cols, data = [], [], []
    for word_idx, neighbors in co_occurrence_dict.items():
        for neighbor_idx, count in neighbors.items():
            rows.append(word_idx)
            cols.append(neighbor_idx)
            data.append(count)

    co_occurrence_matrix = sp.csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.float64)

    # Save matrix with window size in filename
    matrix_filename = f"{output_matrix}_win{window_size}.npz"
    vocab_filename = f"{output_vocab}_win{window_size}.npy"

    sp.save_npz(matrix_filename, co_occurrence_matrix)
    np.save(vocab_filename, vocab)

    print(f"Co-occurrence matrix saved! Shape: {co_occurrence_matrix.shape}, Non-zero values: {co_occurrence_matrix.nnz}")

if __name__ == "__main__":
    # Allow setting window size from the command line
    window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Building co-occurrence matrix with window size = {window_size}")
    build_cooccurrence_matrix("cleaned_corpus.txt", "cooccurrence_matrix", "vocab", window_size)
