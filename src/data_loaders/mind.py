import os
import pandas as pd

from .blueprint_datamodel import DataModel


class MindData(DataModel):
    def __init__(self, root='data/mind', specimen='large', split='train', nrows=None):
        assert specimen in ['small', 'large']
        assert split in ['train', 'test', 'validation', '']

        self.root = os.path.join(root, specimen, split)

        self.behaviours = pd.read_csv(
            os.path.join(self.root, 'behaviors.tsv'), delimiter='\t', nrows=nrows,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions'])
        self.news = pd.read_csv(
            os.path.join(self.root, 'news.tsv'), delimiter='\t',
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
