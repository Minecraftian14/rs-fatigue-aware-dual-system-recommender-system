import os
import pandas as pd

from .blueprint_datamodel import DataModel


class MovieLensData100k(DataModel):
    def __init__(self, root='data/ml-100k'):
        self.root = root

        print(os.getcwd())

        self.users = pd.read_csv(os.path.join(root, 'u.user'), delimiter='|',
                                 names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                                 encoding='latin')
        self.items = pd.read_csv(os.path.join(root, 'u.item'), delimiter='|',
                                 names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
                                        'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                                 encoding='latin')
        self.ratings = pd.read_csv(os.path.join(root, 'u.data'), delimiter='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                   encoding='latin')
        # self.genre = pd.read_csv(os.path.join(root, 'u.genre'), delimiter='\t')
        # self.info = pd.read_csv(os.path.join(root, 'u.info'), delimiter='\t')
        # self.occupation = pd.read_csv(os.path.join(root, 'u.occupation'), delimiter='\t')
