import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data only if not already installed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_corpus(input_folder, output_file):
    """Preprocess text corpus: lowercase, remove punctuation, tokenize, and remove stopwords."""

    sentence_files = [f for f in os.listdir(input_folder) if f.endswith("-sentences.txt")]

    if not sentence_files:
        print("No sentence files found! Check the folder path.")
        print("Folder contents:", os.listdir(input_folder))  # Debugging step
        return
    
    print(f"Found files: {sentence_files}")

    # Load stopwords as a **set** (MUCH FASTER than a list)
    stop_words = set(stopwords.words("english"))

    # Process each file efficiently
    total_sentences = 0
    with open(output_file, "w", encoding="utf-8") as output_f:
        for file in sentence_files:
            file_path = os.path.join(input_folder, file)
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    sentence = parts[-1] if len(parts) > 1 else parts[0]  # Extract correct column
                    
                    # Preprocess the sentence
                    sentence = sentence.lower()
                    sentence = re.sub(r"[^\w\s]", "", sentence)  # Remove punctuation
                    words = word_tokenize(sentence)
                    words = [word for word in words if word not in stop_words]  # Faster stopword removal

                    # Write only non-empty sentences to the output file
                    if words:
                        output_f.write(" ".join(words) + "\n")
                        total_sentences += 1

    print(f"Preprocessing complete! {total_sentences} sentences saved in '{output_file}'.")

#  Run script 
if __name__ == "__main__":
    input_folder = r"C:\Users\bavi0\OneDrive\Documents\assignemnt-precog\eng_news_2024_300K"
    
    preprocess_corpus(input_folder, "cleaned_corpus.txt")
