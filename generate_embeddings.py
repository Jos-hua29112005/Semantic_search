import os

# Force HuggingFace to use local cache
os.environ["HF_HOME"] = r"hf_cache"

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from load_data import documents

# Ensure directories exist
os.makedirs("hf_cache", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(r"model\all_min")

print("Generating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=64,           # faster processing
    show_progress_bar=True
)

# Save embeddings
np.save("embeddings/doc_embeddings.npy", embeddings)

# Save documents for retrieval
with open("embeddings/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Embeddings saved successfully.")
print("Embedding shape:", embeddings.shape)