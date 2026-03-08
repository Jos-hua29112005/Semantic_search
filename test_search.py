import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer(r"D:\Semantic\model\all_min")

# Load FAISS index
index = faiss.read_index("embeddings/faiss_index")

# Load documents
with open("embeddings/documents.pkl", "rb") as f:
    documents = pickle.load(f)

query = "Why do rockets fail during launch?"

print("Query:", query)

query_embedding = model.encode([query])

D, I = index.search(query_embedding, 5)

print("\nTop Results:\n")

for i in I[0]:
    print("---------------")
    print(documents[i][:500])