import numpy as np
import pickle

cluster_probs = np.load("embeddings/cluster_probs.npy")

with open("embeddings/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Boundary documents:\n")

for i, probs in enumerate(cluster_probs):

    sorted_probs = np.sort(probs)[::-1]

    if sorted_probs[0] - sorted_probs[1] < 0.05:

        print("\nDocument:", i)
        print("Top probabilities:", sorted_probs[:3])

        print("\nText sample:\n")
        print(documents[i][:400])