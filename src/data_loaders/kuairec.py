import os
from dataclasses import dataclass

import pandas as pd

from .blueprint_datamodel import DataModel


@dataclass
class DataConfig:
    dataset_name: str = "KuaiRec"
    data_path: str = "./data/KuaiRec2.0/data/"

    # interaction settings
    max_history_length: int = 10
    min_interactions_per_user: int = 5

    # preprocessing
    train_split: float = 0.8
    val_split: float = 0.1

    # filtering
    max_items: int = 5000


class KuaiRecData(DataModel):
    def __init__(self, root='data/KuaiRec2.0/data', specimen='big'):
        assert specimen in ['small', 'big']

        self.root = root

        self.matrix = pd.read_csv(os.path.join(self.root, f'{specimen}_matrix.csv'), )
        self.matrix.sort_values(['user_id', 'timestamp'])
