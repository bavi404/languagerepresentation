# **Cross-Lingual Word Embedding Alignment & Bias Analysis**

## **📌 Project Overview**
This project explores word embeddings by:
- Constructing co-occurrence-based embeddings and comparing them with **Word2Vec, GloVe, and FastText**.
- Aligning **English-Hindi embeddings** using **Procrustes Analysis**.
- Evaluating word embeddings using **SimLex-999, WordSim-353**.
- Analyzing **harmful biases** in embeddings and contextual models (e.g., BERT).

---

## **📁 Directory Structure**
```
📂 assignment-precog
│-- 📂 data/                    # Corpus and bilingual dictionary
│   │-- cleaned_corpus.txt       # Processed English text corpus
│   │-- en-hi.txt                # Bilingual dictionary (English-Hindi word pairs)
│   │-- SimLex-999.txt           # Word similarity dataset
│   │-- wordsim353/              # Word similarity dataset (WordSim-353)
│-- 📂 embeddings/               # Pre-trained word embeddings
│   │-- glove.6B.300d.txt        # GloVe embeddings (English)
│   │-- cc.en.300.vec            # FastText embeddings (English)
│   │-- cc.hi.300.vec            # FastText embeddings (Hindi)
│-- preprocess.py                # Cleans and tokenizes the text corpus
│-- cooccurrence.py              # Constructs the co-occurrence matrix
│-- svd_reduce.py                # Applies SVD for dimensionality reduction
│-- evaluate.py                  # Evaluates embeddings using word similarity datasets
│-- compare_with_glove.py        # Compares co-occurrence embeddings with GloVe
│-- compare_with_fast.py         # Compares co-occurrence embeddings with FastText
│-- alignment.py                 # Aligns English and Hindi embeddings
│-- bias_analysis.py             # Analyzes bias in word embeddings and BERT
│-- README.md                    # Documentation and setup instructions
```

---

## **🛠️ Installation & Dependencies**

### **🔹 Required Libraries:**
Run the following command to install the necessary dependencies:
```bash
pip install numpy pandas scikit-learn scipy gensim nltk matplotlib tqdm
```

Additional dependencies (if using contextual models like BERT):
```bash
pip install transformers torch
```

---

## **🚀 How to Run the Project**

### **1️⃣ Preprocess the Text Corpus**
```bash
python preprocess.py
```
This generates `cleaned_corpus.txt`, which is used for training word embeddings.

### **2️⃣ Construct Word Embeddings**
#### **(A) Co-occurrence Matrix & SVD Reduction**
```bash
python cooccurrence.py
python svd_reduce.py
```
#### **(B) Neural Embeddings (GloVe, FastText)**
Pre-trained embeddings are already provided (`glove.6B.300d.txt`, `cc.en.300.vec`, `cc.hi.300.vec`).

### **3️⃣ Evaluate Word Embeddings**
```bash
python evaluate.py
```
This computes cosine similarities, word clusters, and correlation with **SimLex-999** & **WordSim-353**.

### **4️⃣ Compare Co-occurrence vs. Neural Embeddings**
```bash
python compare_with_glove.py
python compare_with_fast.py
```

### **5️⃣ Align English-Hindi Embeddings**
```bash
python alignment.py
```
This applies **Procrustes Analysis** to align embeddings and evaluates word translation accuracy.

### **6️⃣ Analyze Bias in Embeddings**
```bash
python bias_analysis.py
```
This detects gender/occupation biases in **GloVe, FastText, and BERT** models.

---

## **📊 Methodology**

### **🔹 Word Embedding Construction**
1. Processed a **300K-sentence English corpus**.
2. Built a **co-occurrence matrix** (window sizes: 2, 5, 10).
3. Reduced dimensionality using **Singular Value Decomposition (SVD)**.
4. Compared with **pre-trained embeddings (GloVe, FastText, Word2Vec)**.

### **🔹 Evaluation of Word Embeddings**
- Used **SimLex-999 & WordSim-353** for similarity testing.
- Visualized embeddings using **t-SNE & PCA clustering**.

### **🔹 Cross-Lingual Alignment**
- Used **Procrustes Analysis** to align **English & Hindi embeddings**.
- Evaluated alignment using **word translation retrieval accuracy**.

### **🔹 Bias Analysis**
- Measured gender bias in word embeddings (**e.g., doctor → male bias**).
- Used **BERT Masked Language Modeling (MLM)** to test contextual biases.

---

## **📌 Results Summary**
### **1️⃣ Word Embeddings Performance (Word Similarity Correlation)**
| Method | SimLex-999 | WordSim-353 |
|--------|-----------|-------------|
| Co-occurrence (Window=5) | 0.044 | 0.254 |
| GloVe (300d) | 0.389 | 0.603 |
| FastText | 0.412 | 0.671 |

### **2️⃣ Cross-Lingual Alignment (Procrustes Analysis)**
| Method | Word Translation Accuracy |
|--------|------------------|
| Procrustes Analysis | 52.8% |

### **3️⃣ Bias Detection Results**
| Word | Cosine Similarity (Male) | Cosine Similarity (Female) |
|------|--------------------------|--------------------------|
| Doctor | 0.34 | -0.21 |
| Engineer | 0.42 | -0.15 |
| Nurse | -0.47 | 0.50 |

**Findings:**  
✔ **GloVe/FastText show gender bias (doctor → male, nurse → female).**  
✔ **BERT reinforces gender stereotypes through MLM predictions.**

---

## **📌 Next Steps & Future Work**
- **Improve cross-lingual alignment** with larger bilingual dictionaries.
- **Explore iterative Procrustes refinement** for better accuracy.
- **Mitigate bias using debiasing techniques** (Bolukbasi et al.).
- **Use embeddings in downstream NLP tasks** (e.g., sentiment analysis).

---

## **📚 References**
- Conneau, A., Lample, G., Ranzato, M., Denoyer, L., & Jégou, H. (2018). *Word Translation Without Parallel Data.*
- Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings.*
- Wang, A., et al. (2019). *SuperGLUE: A Stickier Benchmark for NLP Understanding Systems.*

---

### **📌 Acknowledgments**
This project was developed as part of **NLP research on cross-lingual word alignment and bias detection**.

