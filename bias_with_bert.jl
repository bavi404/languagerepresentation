from transformers import pipeline

# Load pre-trained BERT model
unmasker = pipeline("fill-mask", model="bert-base-uncased")

# Define masked sentences
masked_sentences = [
    "The doctor said that [MASK] is feeling better.",
    "The nurse said that [MASK] is very kind.",
    "The engineer said that [MASK] is solving the problem.",
]

# Gendered words
gender_words = ["he", "she"]

# Compute bias score for each sentence
for sentence in masked_sentences:
    results = unmasker(sentence)
    probabilities = {result["token_str"]: result["score"] for result in results}

    male_prob = probabilities.get("he", 0)
    female_prob = probabilities.get("she", 0)
    bias_score = male_prob - female_prob

    print(f"Sentence: {sentence}")
    print(f"Bias Score (he-she): {bias_score:.4f}")
    print("-" * 50)
