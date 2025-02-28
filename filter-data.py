import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_corpus(input_folder, output_file):
    sentence_files = [f for f in os.listdir(input_folder) if f.endswith("-sentences.txt")]
    cleaned_sentences = []

    for file in sentence_files:
        file_path = os.path.join(input_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip().split("\t")[-1]
                sentence = sentence.lower()
                sentence = re.sub(r"[^\w\s]", "", sentence)
                words = word_tokenize(sentence)
                words = [word for word in words if word not in stopwords.words("english")]
                if words:
                    cleaned_sentences.append(" ".join(words))

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in cleaned_sentences:
            f.write(sentence + "\n")

    print(f"Preprocessing complete! {len(cleaned_sentences)} sentences saved.")

# Run if executed directly
if __name__ == "__main__":
    preprocess_corpus("path/to/extracted/folder", "cleaned_corpus.txt")
