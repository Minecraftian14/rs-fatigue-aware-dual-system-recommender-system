import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from miscellaneous import smart_encode, balanced_subset

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
    def __init__(self, root='data/KuaiRec2.0', initial_specimen='small'):
        self.root = root
        self.data_root = os.path.join(self.root, 'data')

        # item_daily_features = pd.read_csv(
        #     os.path.join(self.data_root, 'item_daily_features.csv'), )
        # social_network = pd.read_csv(
        #     os.path.join(self.data_root, 'social_network.csv'), )
        self.user_features = self._load_user_features()
        self.item_features = self._load_item_features()

        self.ratings: pd.DataFrame = None
        self.big_ratings: pd.DataFrame = None
        self.small_ratings: pd.DataFrame = None
        self.smaller_ratings: pd.DataFrame = None
        self.load_specimen(initial_specimen)

        self.n_users, self.d_user_feats = self.user_features.shape
        self.n_items, self.d_item_feats = self.item_features.shape

    def _load_user_features(self):
        cache_file = os.path.join(self.data_root, 'user_features_processed.parquet')
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        user_features = pd.read_csv(os.path.join(self.data_root, 'user_features.csv'))
        user_features.drop(columns=[
            'is_lowactive_period', 'is_live_streamer', 'is_video_author',
            'onehot_feat0', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
            'onehot_feat15', 'onehot_feat16', 'onehot_feat17',
        ], inplace=True)

        extracted_features = smart_encode(user_features, ignore_cols={
            'user_id',
            # Allow for categorical for the time being
            # 'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range',  # Ranges
        }, force_ohe={
            f"onehot_feat{x}" for x in range(0, 18)
        }, index=user_features['user_id'])

        extracted_features['follow_user_num_range_log_of_mid'] = user_features['follow_user_num_range'].map(
            {'0': 0, '(0,10]': np.log((0 + 10) // 2), '(10,50]': np.log((10 + 50) // 2), '(50,100]': np.log((50 + 100) // 2),
             '(100,150]': np.log((100 + 150) // 2), '(150,250]': np.log((150 + 250) // 2), '(250,500]': np.log((250 + 500) // 2), '500+': np.log(500), })
        extracted_features['fans_user_num_range_log_of_mid'] = user_features['fans_user_num_range'].map({
            '0': 0, '[1,10)': np.log((1 + 10) // 2), '[10,100)': np.log((10 + 100) // 2), '[100,1k)': np.log((100 + 1000) // 2),
            '[1k,5k)': np.log((1000 + 5000) // 2), '[5k,1w)': np.log((5000 + 10000) // 2), '[1w,10w)': np.log((10000 + 100000) // 2), })
        extracted_features['friend_user_num_range_log_of_mid'] = user_features['friend_user_num_range'].map({
            '0': 0, '[1,5)': np.log((1 + 10) // 2), '[5,30)': np.log((5 + 30) // 2), '[30,60)': np.log((30 + 60) // 2),
            '[60,120)': np.log((60 + 120) // 2), '[120,250)': np.log((120 + 250) // 2), '250+': np.log(250), })
        extracted_features['register_days_range_log_of_mid'] = user_features['register_days_range'].map({
            '15-30': np.log((15 + 30) // 2), '31-60': np.log((31 + 60) // 2), '61-90': np.log((61 + 90) // 2), '91-180': np.log((91 + 180) // 2),
            '181-365': np.log((181 + 365) // 2), '366-730': np.log((366 + 730) // 2), '730+': np.log(730), })

        for col in extracted_features.columns:
            column = extracted_features[col]
            if column.min() == 0.0 and column.max() == 1.0: continue
            if pd.api.types.is_bool_dtype(column): continue
            extracted_features[col] = (column - column.mean()) / column.std()

        del user_features
        extracted_features.to_parquet(cache_file)
        return extracted_features

    def _load_item_features(self):
        cache_file = os.path.join(self.data_root, 'item_features_processed.parquet')
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        item_categories = pd.read_csv(os.path.join(self.data_root, 'item_categories.csv'))
        kuairec_caption_category = pd.read_csv(os.path.join(self.data_root, 'kuairec_caption_category.csv'), engine='python')
        assert len(item_categories) == len(kuairec_caption_category)
        item_categories.sort_values(by=['video_id'], inplace=True)
        kuairec_caption_category.sort_values(by=['video_id'], inplace=True)

        extracted_features = smart_encode(kuairec_caption_category, ignore_cols={
            'video_id',
            'manual_cover_text', 'caption', 'topic_tag',  # NLP Features
            'first_level_category_id', 'second_level_category_id', 'third_level_category_id',  # Duplicated because of name
        }, index=kuairec_caption_category['video_id'])

        feat = item_categories['feat'].map(lambda x: [int(y) for y in x.strip()[1:-1].split(', ')])
        feat_min, feat_max = feat.map(lambda x: min(x)).min(), feat.map(lambda x: max(x)).max()
        for i in range(feat_min, feat_max + 1):
            extracted_features[f'feat_{i}'] = feat.map(lambda x: i in x)

        for col in extracted_features.columns:
            column = extracted_features[col]
            if column.min() == 0.0 and column.max() == 1.0: continue
            if pd.api.types.is_bool_dtype(column): continue
            extracted_features[col] = (column - column.mean()) / column.std()

        del item_categories, kuairec_caption_category
        extracted_features.to_parquet(cache_file)
        return extracted_features

    def load_specimen(self, initial_specimen):
        assert initial_specimen in {'big', 'small', 'smaller'}
        if initial_specimen == 'big':
            self.ratings = self.big_ratings = self._load_big_ratings()
        elif initial_specimen == 'small':
            self.ratings = self.small_ratings = self._load_small_ratings()
        elif initial_specimen == 'smaller':
            self.ratings = self.smaller_ratings = self._load_smaller_ratings()

    def _load_big_ratings(self):
        cache_file = os.path.join(self.data_root, 'big_ratings_processed.parquet')
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        big_matrix = pd.read_csv(os.path.join(self.data_root, 'big_matrix.csv'))
        big_matrix = big_matrix[big_matrix['watch_ratio'] <= 10]
        extracted_features = big_matrix[['user_id', 'video_id', 'watch_ratio', 'time']]
        extracted_features.rename(columns={
            'video_id': 'item_id',
            'watch_ratio': 'rating',
        }, inplace=True)

        del big_matrix
        extracted_features.to_parquet(cache_file)
        return extracted_features

    def _load_small_ratings(self):
        cache_file = os.path.join(self.data_root, 'small_ratings_processed.parquet')
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        small_matrix = pd.read_csv(os.path.join(self.data_root, 'small_matrix.csv'))
        small_matrix = small_matrix[small_matrix['watch_ratio'] <= 10]
        extracted_features = small_matrix[['user_id', 'video_id', 'watch_ratio', 'time']]
        extracted_features.rename(columns={
            'video_id': 'item_id',
            'watch_ratio': 'rating',
        }, inplace=True)

        del small_matrix
        extracted_features.to_parquet(cache_file)
        return extracted_features

    def _load_smaller_ratings(self):
        cache_file = os.path.join(self.data_root, 'smaller_ratings_processed.parquet')
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        small_matrix = pd.read_csv(os.path.join(self.data_root, 'small_matrix.csv'))
        small_matrix = small_matrix[small_matrix['watch_ratio'] <= 10]
        extracted_features = small_matrix[['user_id', 'video_id', 'watch_ratio', 'time']]
        smaller_matrix = balanced_subset(extracted_features, 1000000)
        smaller_matrix.rename(columns={
            'video_id': 'item_id',
            'watch_ratio': 'rating',
        }, inplace=True)

        del small_matrix, extracted_features
        smaller_matrix.to_parquet(cache_file)
        return smaller_matrix

    def generate_embeddings(self):
        ...
