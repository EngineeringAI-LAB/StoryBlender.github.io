from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

MODEL_CACHE_DIR = str(Path(__file__).resolve().parent / "bge_small_en_v1.5_model")

def cosine_similarity(a, b):
    """
    Helper function to calculate the similarity between two vectors.
    Returns a score between 0 (not similar) and 1 (identical).
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 1. Initialize the Model
# This automatically downloads the quantized BAAI/bge-small-en-v1.5 model
# It is extremely fast and lightweight.
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=MODEL_CACHE_DIR, local_files_only=True)
print("The model BAAI/bge-small-en-v1.5 model is ready to use.")
print("-" * 50)

# 2. Define your inputs
user_query = "Dimly lit stone chamber with flickering candlelight ambiance."
candidates_list = [
    "Free unclipped 16K HDRI of an abandoned church in autumn - soft, low-contrast natural morning/afternoon light under partly cloudy skies.",
    "Free, unclipped 16K HDRI of an abandoned brick warehouse interior - soft overcast natural light, medium contrast, desolate, dusty, rubble-filled atmosphere.",
    "Free 16K unclipped HDRI of an abandoned factory canteen - soft natural window light with cool fluorescent fill, low-contrast industrial hall atmosphere.",
    "Unclipped, free 20K HDRI: low-contrast sunrise with partly cloudy sky, gentle warm light and soft ambient reflections for outdoor, natural lighting.",
]

# 3. Generate Embeddings
# Note: model.embed() expects a list (iterator) of strings
# We convert the generator to a list to use it in calculations
print("Encoding query and candidates...")
query_embedding = list(embedding_model.embed([user_query]))[0]
candidate_embeddings = list(embedding_model.embed(candidates_list))

# 4. Search: Calculate Scores
# We loop through candidates and calculate the cosine similarity for each
results = []
for i, candidate_emb in enumerate(candidate_embeddings):
    score = cosine_similarity(query_embedding, candidate_emb)
    results.append((candidates_list[i], score))

# 5. Sort and Display
# Sort by score descending (highest relevance first)
results.sort(key=lambda x: x[1], reverse=True)

print(f"\nQuery: '{user_query}'\n")
print("Relevance Candidates:")
print("-" * 30)
for candidate, score in results:
    print(f"Score: {score:.4f} | {candidate}")