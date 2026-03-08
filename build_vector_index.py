import numpy as np
import faiss

print("Loading embeddings...")

embeddings = np.load("embeddings/doc_embeddings.npy")

dimension = embeddings.shape[1]

print("Creating FAISS index...")

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, "embeddings/faiss_index")

print("FAISS index created successfully.")
print("Total vectors:", index.ntotal)