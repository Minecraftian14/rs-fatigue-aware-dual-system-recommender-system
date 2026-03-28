import numpy as np

from .blueprint_datamodel import DataModel


class SyntheticData(DataModel):
    def __init__(self,
                 n_items=500, d_embeds=16,
                 n_clusters=5, std_cluster=0.1,
                 ):
        super(SyntheticData, self).__init__()
        self.n_items = n_items
        self.d_item_embeds = d_embeds

        centers = np.random.randn(n_clusters, d_embeds)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        probabilities = np.exp(np.linspace(n_clusters * 0.5555555, 1, n_clusters))
        probabilities = probabilities / np.sum(probabilities)
        cluster_ids = np.random.choice(n_clusters, size=n_items, p=probabilities)

        self.embeddings = np.zeros((n_items, d_embeds))

        for i in range(n_items):
            center = centers[cluster_ids[i]]
            noise = np.random.randn(d_embeds) * std_cluster
            vec = center + noise
            self.embeddings[i] = vec / np.linalg.norm(vec)

    def get_n_items(self) -> int:
        return self.n_items

    def get_d_item_embeds(self) -> int:
        return self.d_item_embeds

    def get_items_embeds(self, item_idx: int | slice | None = None) -> np.ndarray:
        if item_idx is None: return self.embeddings
        return self.embeddings[item_idx]
