import numpy as np
import pickle
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from semantic_cache import SemanticCache

# -------------------------------------------------
# Initialize API
# -------------------------------------------------

app = FastAPI()

print("Loading models...")

# Load embedding model
model = SentenceTransformer(r"model/all_min")

# Load FAISS index
index = faiss.read_index("embeddings/faiss_index")

# Load documents
with open("embeddings/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Load clustering model
with open("embeddings/gmm_model.pkl", "rb") as f:
    gmm = pickle.load(f)

# Initialize semantic cache
cache = SemanticCache(threshold=0.85)

print("System ready.")

# -------------------------------------------------
# Request schema
# -------------------------------------------------

class QueryRequest(BaseModel):
    query: str


# -------------------------------------------------
# Query Endpoint
# -------------------------------------------------

@app.post("/query")
def query_endpoint(req: QueryRequest):

    query = req.query

    # Generate query embedding
    embedding = model.encode(query)

    # Predict cluster
    cluster_probs = gmm.predict_proba([embedding])[0]
    dominant_cluster = int(np.argmax(cluster_probs))

    # Check semantic cache
    cached = cache.lookup(embedding, dominant_cluster)

    if cached:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": cached["matched_query"],
            "similarity_score": float(cached["similarity"]),
            "result": cached["result"],
            "dominant_cluster": dominant_cluster
        }

    # If cache miss → perform FAISS search
    D, I = index.search(np.array([embedding]), 5)

    results = [documents[i][:500] for i in I[0]]

    # Store result in cache
    cache.add(query, embedding, dominant_cluster, results)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": dominant_cluster
    }


# -------------------------------------------------
# Cache statistics endpoint
# -------------------------------------------------

@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# -------------------------------------------------
# Clear cache endpoint
# -------------------------------------------------

@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"status": "cache cleared"}