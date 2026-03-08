import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        self.threshold = threshold
        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0


    def lookup(self, query_embedding, cluster):

        if cluster not in self.cache:

            self.miss_count += 1
            return None

        cluster_data = self.cache[cluster]

        embeddings = cluster_data["embeddings"]

        similarities = cosine_similarity(
            [query_embedding],
            embeddings
        )[0]

        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= self.threshold:

            self.hit_count += 1

            return {
                "matched_query": cluster_data["queries"][best_idx],
                "similarity": float(best_sim),
                "result": cluster_data["results"][best_idx]
            }

        self.miss_count += 1
        return None


    def add(self, query, embedding, cluster, result):

        if cluster not in self.cache:

            self.cache[cluster] = {
                "embeddings": np.array([embedding]),
                "queries": [query],
                "results": [result]
            }

        else:

            cluster_data = self.cache[cluster]

            cluster_data["embeddings"] = np.vstack([
                cluster_data["embeddings"],
                embedding
            ])

            cluster_data["queries"].append(query)
            cluster_data["results"].append(result)


    def stats(self):

        total = self.hit_count + self.miss_count

        total_entries = sum(
            len(v["queries"]) for v in self.cache.values()
        )

        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total else 0
        }


    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0