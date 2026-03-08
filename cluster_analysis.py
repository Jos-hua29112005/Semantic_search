import numpy as np
import pickle

print("Loading clustering results...")

cluster_probs = np.load("embeddings/cluster_probs.npy")

with open("embeddings/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Cluster probability matrix shape:", cluster_probs.shape)

n_clusters = cluster_probs.shape[1]

# Dominant cluster for each document
dominant_cluster = np.argmax(cluster_probs, axis=1)

for c in range(n_clusters):

    print("\n==============================")
    print("Cluster", c)
    print("==============================")

    indices = np.where(dominant_cluster == c)[0]

    print("Documents in cluster:", len(indices))

    # Print a few example documents
   