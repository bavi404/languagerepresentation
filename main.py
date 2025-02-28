import os

print("Step 1: Preprocessing Corpus...")
os.system("python preprocess.py")

print("Step 2: Building Co-occurrence Matrix...")
os.system("python cooccurrence.py")

print("Step 3: Applying Truncated SVD...")
os.system("python svd_reduce.py")

print("Step 4: Evaluating Embeddings...")
os.system("python evaluate.py")
