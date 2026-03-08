import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

print("Loading embeddings...")
embeddings = np.load("embeddings/doc_embeddings.npy")

bic_scores = []
cluster_range = range(10, 60, 5)

for k in cluster_range:

    print("Testing clusters:", k)

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=42
    )

    gmm.fit(embeddings)

    bic_scores.append(gmm.bic(embeddings))

# Plot BIC curve
plt.plot(cluster_range, bic_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("BIC Score")
plt.title("Cluster Selection using BIC")
plt.show()

best_k = cluster_range[bic_scores.index(min(bic_scores))]

print("Best cluster count:", best_k)