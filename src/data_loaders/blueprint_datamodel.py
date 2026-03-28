from abc import ABC
import numpy as np


class DataModel(ABC):
    def get_n_items(self) -> int:
        pass

    def get_d_item_embeds(self) -> int:
        pass

    def get_items_embeds(self, item_idx: int | slice | None = None) -> np.ndarray:
        pass
