import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import wget


class Dataset:
    def __init__(self, name: str):
        assert name in ['ml-1m', 'bcr', 'pin'], f'{name} dataset is not supported'
        self.name = name

    def get_data(self) -> pd.DataFrame:
        if self.name == 'ml-1m':
            working_dir = Path('../data/ml-1m')
            if not working_dir.exists():
                working_dir.mkdir(parents=True)

            data_path = working_dir / 'ml-1m/ratings.dat'
            if not data_path.exists():
                data_url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
                wget.download(data_url, str(working_dir))
                with zipfile.ZipFile(working_dir / 'ml-1m.zip', 'r') as zip_ref:
                    zip_ref.extractall(working_dir)

            data = pd.read_csv(
                data_path,
                engine='python',  # c engine doesn't support separators > 1 char
                sep='::',
                header=None,
                names=['user_id', 'item_id', 'rating', 'event_ts']
            )

            data = (
                pl.from_pandas(data)
                .filter(pl.col('rating') >= 4)
                .select(['user_id', 'item_id', 'event_ts'])
            )
            return data
        elif self.name == 'bcr':
            working_dir = Path('../data/bcr')
            if not working_dir.exists():
                working_dir.mkdir(parents=True)

            data_path = working_dir / 'BX-Book-Ratings.csv'
            if not data_path.exists():
                data_url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
                wget.download(data_url, str(working_dir))
                with zipfile.ZipFile(working_dir / 'BX-CSV-Dump.zip', 'r') as zip_ref:
                    zip_ref.extractall(working_dir)

            data = pd.read_csv(
                working_dir / 'BX-Book-Ratings.csv',
                sep=';',
                encoding_errors='ignore'
            )

            data = (
                pl.from_pandas(data)
                .filter(pl.col('Book-Rating') >= 6)
            )

            data = (
                data
                .with_columns([
                    pl.col('User-ID').alias('user_id'),
                    pl.col('ISBN').alias('item_id'),
                    pl.Series(np.arange(len(data))).alias('event_ts')
                ])
                .select(['user_id', 'item_id', 'event_ts'])
            )
            return data
        elif self.name == 'pin':
            working_dir = Path('../data/pin')
            if not working_dir.exists():
                working_dir.mkdir(parents=True)

            data_path = working_dir / 'pinterest.csv'
            if not data_path.exists():
                data_url = 'https://github.com/edervishaj/pinterest-recsys-dataset/raw/main/pinterest.csv'
                wget.download(data_url, str(working_dir))

            data = pl.read_csv('../pinterest.csv')

            data = (
                data
                .with_columns([
                    pl.col('board_id').alias('user_id'),
                    pl.col('image').alias('item_id'),
                    (pl.Series(np.arange(len(data)))).alias('event_ts')
                ])
                .select(['user_id', 'item_id', 'event_ts'])
            )
            return data


if __name__ == '__main__':
    print(Dataset('pin').get_data())
