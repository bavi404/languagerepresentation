import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sparse_linalg
import sys

def apply_svd(matrix_file, vocab_file, output_file, d=300):
    """Applies Truncated SVD to reduce the co-occurrence matrix dimensions."""
    
    # Load matrix with specified window size
    co_occurrence_matrix = sp.load_npz(matrix_file).astype(np.float64)
    vocab = np.load(vocab_file, allow_pickle=True).item()

    print(f"Loaded matrix {matrix_file} of shape {co_occurrence_matrix.shape} with {co_occurrence_matrix.nnz} non-zero values.")

    # Ensure d is valid
    max_d = min(co_occurrence_matrix.shape) - 1
    if d > max_d:
        print(f"Warning: `d`={d} is too large! Reducing to {max_d}.")
        d = max_d

    print(f"Running SVD with `d`={d}...")
    U, S, Vt = sparse_linalg.svds(co_occurrence_matrix, k=d)

    # Compute embeddings: U @ Σ
    word_embeddings = U @ np.diag(S)

    # Save embeddings with window size in filename
    np.save(output_file, word_embeddings)
    print(f"SVD complete! Reduced matrix shape: {word_embeddings.shape}")

if __name__ == "__main__":
    window_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Running SVD for window size {window_size}...")

    apply_svd(f"cooccurrence_matrix_win{window_size}.npz", 
              f"vocab_win{window_size}.npy", 
              f"word_embeddings_win{window_size}.npy")
