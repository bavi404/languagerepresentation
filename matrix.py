import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict

def build_cooccurrence_matrix(input_file, output_matrix, output_vocab, window_size=5):
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f.readlines()]

    word_counts = Counter(word for sentence in sentences for word in sentence)
    vocab = {word: i for i, word in enumerate(word_counts.keys())}
    
    co_occurrence_dict = defaultdict(lambda: defaultdict(int))

    for sentence in sentences:
        for idx, word in enumerate(sentence):
            word_idx = vocab[word]
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size + 1, len(sentence))]:
                if neighbor != word:
                    neighbor_idx = vocab[neighbor]
                    co_occurrence_dict[word_idx][neighbor_idx] += 1

    rows, cols, data = [], [], []
    for word_idx, neighbors in co_occurrence_dict.items():
        for neighbor_idx, count in neighbors.items():
            rows.append(word_idx)
            cols.append(neighbor_idx)
            data.append(count)

    co_occurrence_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=np.float64)

    sp.save_npz(output_matrix, co_occurrence_matrix)
    np.save(output_vocab, vocab)

    print(f"Co-occurrence matrix saved! Shape: {co_occurrence_matrix.shape}")

if __name__ == "__main__":
    build_cooccurrence_matrix("cleaned_corpus.txt", "cooccurrence_matrix.npz", "vocab.npy")
