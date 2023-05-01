from collections import defaultdict
from typing import Dict, NoReturn, Set, Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


class IdsEncoder:
    def __init__(self):
        self.user_id_mapping, self.user_id_inverse_mapping = None, None
        self.item_id_mapping, self.item_id_inverse_mapping = None, None

    def fit(self, data: pl.DataFrame) -> NoReturn:
        self.user_id_mapping = {k: v for v, k in enumerate(data['user_id'].unique().to_numpy())}
        self.user_id_inverse_mapping = {k: v for v, k in self.user_id_mapping.items()}

        self.item_id_mapping = {k: v for v, k in enumerate(data['item_id'].unique().to_numpy())}
        self.item_id_inverse_mapping = {k: v for v, k in self.item_id_mapping.items()}

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        assert self.user_id_mapping is not None
        assert self.item_id_mapping is not None
        assert 'user_id' in data.columns
        assert 'item_id' in data.columns

        return (
            data
            .with_columns([
                pl.col('user_id').apply(self.user_id_mapping.get).alias('user_id'),
                pl.col('item_id').apply(self.item_id_mapping.get).alias('item_id')
            ])
        )

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        assert self.user_id_inverse_mapping is not None
        assert self.item_id_inverse_mapping is not None
        assert 'user_id' in data.columns
        assert 'item_id' in data.columns

        return (
            data
            .with_columns([
                pl.col('user_id').apply(self.user_id_inverse_mapping.get).alias('user_id'),
                pl.col('item_id').apply(self.item_id_inverse_mapping.get).alias('item_id')
            ])
        )


class MinInteractionsFilter:
    def __init__(self, min_user_interactions: int = 5, min_item_interactions: int = 5):
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        assert 'user_id' in data.columns
        assert 'item_id' in data.columns

        filtered_items = (
            data
            .groupby('item_id')
            .count()
            .filter(pl.col('count') >= self.min_item_interactions)
            .select('item_id')
        )
        data = data.join(filtered_items, on='item_id')

        filtered_users = (
            data
            .groupby('user_id')
            .count()
            .filter(pl.col('count') >= self.min_user_interactions)
            .select('user_id')
        )
        data = data.join(filtered_users, on='user_id')

        return data


class SplitTrainValTest:
    def __init__(self, train_size: float = 0.6, val_size: float = 0.2):
        assert train_size + val_size <= 1.0
        self.train_size = train_size
        self.val_size = val_size

    def transform(self, data: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if 'event_ts' in data.columns:
            # split like time series
            train_val_ts_threshold = np.quantile(data.select('event_ts'), self.train_size)
            val_test_ts_threshold = np.quantile(data.select('event_ts'), self.train_size + self.val_size)
            train_df = data.filter(pl.col('event_ts') <= train_val_ts_threshold)
            val_df = data.filter(
                (pl.col('event_ts') > train_val_ts_threshold) &
                (pl.col('event_ts') <= val_test_ts_threshold)
            )
            test_df = data.filter(pl.col('event_ts') > val_test_ts_threshold)
            return train_df, val_df, test_df

        # split like sequences
        train_seqs = defaultdict(list)
        val_seqs = defaultdict(list)
        test_seqs = defaultdict(list)
        for user_id, item_ids in data.groupby('user_id').agg(pl.col('item_id')).rows():
            train_seq = item_ids[:int(len(item_ids) * self.train_size)]
            val_seq = item_ids[len(train_seq):len(train_seq) + int(len(item_ids) * self.val_size)]
            test_seq = item_ids[len(train_seq) + len(val_seq):]

            train_seqs['user_id'].append(user_id)
            train_seqs['item_id'].append(train_seq)

            val_seqs['user_id'].append(user_id)
            val_seqs['item_id'].append(val_seq)

            test_seqs['user_id'].append(user_id)
            test_seqs['item_id'].append(test_seq)
        return (
            pl.from_dict(train_seqs).explode('item_id'),
            pl.from_dict(val_seqs).explode('item_id'),
            pl.from_dict(test_seqs).explode('item_id')
        )


def create_sparse_dataset(data: pl.DataFrame, shape=None) -> csr_matrix:
    if shape is None:
        shape = (data['user_id'].max() + 1, data['item_id'].max() + 1)
    return csr_matrix((np.ones(len(data)), (data['user_id'], data['item_id'])), shape=shape)


def create_positives_dataset(data: pl.DataFrame, apply_set: bool = True) -> Dict[int, Set[int]]:
    positives = data.groupby('user_id')

    if 'event_ts' in data.columns:
        positives = positives.agg(pl.col('item_id').sort_by('event_ts'))
    else:
        positives = positives.agg(pl.col('item_id'))
    positives = positives.to_dict(as_series=False)

    return {
        k: set(v) if apply_set else list(v)
        for k, v in zip(positives['user_id'], positives['item_id'])
    }
