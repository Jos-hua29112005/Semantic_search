import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

print("Loading embeddings...")

embeddings = np.load("embeddings/doc_embeddings.npy")

print("Embedding shape:", embeddings.shape)

# -------------------------------------------------
# Number of clusters
# -------------------------------------------------
N_CLUSTERS = 30

print("Training Gaussian Mixture Model...")

gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="diag",
    random_state=42,
    max_iter=200
)

gmm.fit(embeddings)

print("Model training completed.")

# -------------------------------------------------
# Get cluster probabilities
# -------------------------------------------------
print("Computing cluster membership probabilities...")

cluster_probs = gmm.predict_proba(embeddings)

print("Cluster probability matrix shape:", cluster_probs.shape)

# -------------------------------------------------
# Save clustering results
# -------------------------------------------------
np.save("embeddings/cluster_probs.npy", cluster_probs)

with open("embeddings/gmm_model.pkl", "wb") as f:
    pickle.dump(gmm, f)

print("Clustering results saved.")