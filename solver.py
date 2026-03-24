import gensim.downloader as api
import numpy as np

# Load the pre-trained Word2Vec model
# Note: This code is commented out because it requires downloading a large model from the internet.
#print("Loading model...")
#model = api.load("word2vec-google-news-300")
#print("Model loaded!")
#model.save("word2vec.model")

# Load the model from the saved file
from gensim.models import KeyedVectors
#print("Loading model from file...")
model = KeyedVectors.load("word2vec.model")
print("Model loaded!")

# Compute cosine similarity between two vectors.
# Returns a float in [-1, 1], or 0 if either vector has zero magnitude.
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# Get the top 60,000 most common words in the model's vocabulary
vocab = model.index_to_key[:60000]

# Pre-compute and cache all word vectors for fast repeated lookups.
# Storing them in a dict avoids redundant model lookups during filtering.
vectors = {word: model[word] for word in vocab}


def filter_candidates(guess, score, candidates, tol=0.05):
    """
    Filter candidate words to those whose cosine similarity with `guess`
    is within `tol` of the target `score`.

    Args:
        guess      (str):        The guessed word to compare against.
        score      (float):      Target cosine similarity (e.g. similarity
                                 between guess and the secret target word).
        candidates (list[str]):  Pool of words to filter down.
        tol        (float):      Tolerance window around `score`.
                                 A candidate is kept if:
                                     |cosine(guess, candidate) - score| <= tol

    Returns:
        list[str]: Filtered candidates within the similarity window.
    """
    # Guard: if the guess isn't in the model, we can't compute similarities.
    if guess not in vectors:
        print(f"Warning: '{guess}' not found in vocabulary. Returning all candidates unchanged.")
        return list(candidates)

    guess_vec = vectors[guess]           # Retrieve the guess vector once
    filtered  = []                       # Accumulate matching candidates

    for word in candidates:
        # Skip the guess word itself — it's trivially similar (score = 1.0)
        if word == guess:
            continue

        # Skip any candidate not present in our pre-cached vector dict
        if word not in vectors:
            continue

        candidate_vec = vectors[word]

        # Cosine similarity between the guess and this candidate
        similarity = cosine(guess_vec, candidate_vec)

        # Keep the candidate if it falls within [score - tol, score + tol]
        if abs(similarity - score) <= tol:
            filtered.append(word)

    return filtered


# ── Example usage ────────────────────────────────────────────────────────────
target    = "tiger"
guess     = "animal"
score     = cosine(model[guess], model[target])   # similarity of guess → target

print(f"Cosine similarity between '{guess}' and '{target}': {score:.4f}")

candidates = list(vocab)                          # start with the full vocab
filtered   = filter_candidates(guess, score, candidates)

print(f"Candidates remaining: {len(filtered)}")
print("Sample:", filtered[:10])