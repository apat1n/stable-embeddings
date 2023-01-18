import datetime
import random
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import plotly.express as px
from implicit.bpr import BayesianPersonalizedRanking

from dataset import Dataset
from metrics import user_recall
from postprocess_data import (IdsEncoder, MinInteractionsFilter,
                              SplitTrainValTest, create_positives_dataset,
                              create_sparse_dataset)

SEED = 42
DIM = 128
LR_START = 1e-1
LR_END = 1e-3
EPOCHS = 100
RESTART_EPOCHS = 100
REG_FACTOR = 1e-2


def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)


def get_model():
    return BayesianPersonalizedRanking(
        iterations=EPOCHS, factors=(DIM - 1), random_state=SEED,
        learning_rate=LR_START, regularization=REG_FACTOR
    )


def linear_lr(epoch, n_epochs, lr_start, lr_end):
    p = epoch / n_epochs
    return lr_start * (1 - p) + lr_end * p


def cosine_annealing_warm_restart(epoch, n_epochs, lr_start, lr_end, t_i):
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    t_cur = epoch % t_i
    # modification to make lr after restart closer to lr_end
    linear_coef = (n_epochs - epoch) / n_epochs
    return lr_end + linear_coef * (lr_start - lr_end) * (1 + np.cos(np.pi * t_cur / t_i)) / 2


class CallbackClass:
    def __init__(self, model, train_positives, val_positives):
        self.model = model
        self.train_positives = train_positives
        self.val_positives = val_positives

        self.prev_user_factors = None
        self.prev_item_factors = None
        self.prev_recommendations = defaultdict(lambda: None)

        self.user_bias_history = []
        self.item_bias_history = []
        self.recommendation_intersection_history = defaultdict(list)
        self.recall_history = defaultdict(list)

    def callback_fn(self, epoch, *_):
        self.model.learning_rate = linear_lr(epoch, self.model.iterations, LR_START, LR_END)

        if self.prev_user_factors is not None:
            user_factors_bias = np.abs(self.prev_user_factors - self.model.user_factors).mean(axis=1).mean()
            self.user_bias_history.append(user_factors_bias)
        self.prev_user_factors = self.model.user_factors.copy()

        if self.prev_item_factors is not None:
            item_factors_bias = np.abs(self.prev_item_factors - self.model.item_factors).mean(axis=1).mean()
            self.item_bias_history.append(item_factors_bias)
        self.prev_item_factors = self.model.item_factors.copy()

        index = faiss.IndexFlatIP(DIM)
        index.add(self.model.item_factors)
        recs = index.search(self.model.user_factors, 40)[1]

        for k in [1, 10, 20]:
            recall_list = []
            for user_id, y_true in self.val_positives.items():
                y_pred = [
                    item_id for item_id in recs[user_id]
                    if item_id not in self.train_positives.get(user_id, set())
                ]
                recall_list.append(user_recall(y_pred, y_true, k))
            self.recall_history[k].append(np.mean(recall_list))

            if self.prev_recommendations[k] is not None:
                intersection_count = []
                for prev_recs, cur_recs in zip(
                        self.prev_recommendations[k][list(self.train_positives.keys())],
                        recs[list(self.train_positives.keys())]
                ):
                    intersection_count.append(len(set(cur_recs[:k]).intersection(prev_recs)))
                self.recommendation_intersection_history[k].append(np.mean(intersection_count) / k)

            self.prev_recommendations[k] = recs


def main(dataset_name: str = 'ml-1m', verbose: bool = True):
    data = Dataset(dataset_name).get_data()

    min_iterations_filter = MinInteractionsFilter()
    ids_encoder = IdsEncoder()

    data = min_iterations_filter.transform(data)
    data = ids_encoder.fit_transform(data)

    n_users = len(data['user_id'].unique())
    n_items = len(data['item_id'].unique())
    n_interactions = len(data)
    print(f'users: {n_users}')
    print(f'items: {n_items}')
    print(f'interactions: {n_interactions}')
    print(f'density: {(n_interactions / n_users / n_items) * 100:.2f}%')

    splitter = SplitTrainValTest()
    train_df, val_df, test_df = splitter.transform(data)

    train_dataset = create_sparse_dataset(train_df)
    train_positives = create_positives_dataset(train_df)
    val_positives = create_positives_dataset(val_df)
    test_positives = create_positives_dataset(test_df)

    print(f'{len(train_positives.keys())} users in train')
    print(f'{len(val_positives.keys())} users in val')
    print(f'{len(test_positives.keys())} users in test')

    set_seed()
    model = get_model()
    fit_callback = CallbackClass(model, train_positives, val_positives)
    model.fit(train_dataset, callback=fit_callback.callback_fn, show_progress=verbose)

    workdir = Path(f'logs/{datetime.datetime.now()}')
    if not workdir.exists():
        workdir.mkdir(parents=True)

    df = pd.concat([
        pd.DataFrame({
            'epoch': np.arange(len(v)) + 2,
            'recall': np.array(v),
            'k': [k for _ in v]
        })
        for k, v in fit_callback.recall_history.items()
    ])
    fig = px.line(
        df, x="epoch", y="recall", color='k',
        title='recall@k on validation set'
    )
    if verbose:
        fig.show()
    fig.write_image(workdir / 'recall.png')

    df = pd.concat([
        pd.DataFrame({
            'epoch': np.arange(len(fit_callback.user_bias_history)) + 2,
            'bias': fit_callback.user_bias_history,
            'embedding_type': ['user_emb' for _ in fit_callback.user_bias_history]
        }),
        pd.DataFrame({
            'epoch': np.arange(len(fit_callback.item_bias_history)) + 2,
            'bias': fit_callback.item_bias_history,
            'embedding_type': ['item_emb' for _ in fit_callback.item_bias_history]
        })
    ])
    fig = px.line(
        df, x="epoch", y="bias", color='embedding_type',
        title='user embeddings bias between successive epochs'
    )
    if verbose:
        fig.show()
    fig.write_image(workdir / 'embeddings_diff.png')

    df = pd.concat([
        pd.DataFrame({
            'epoch': np.arange(len(v)) + 2,
            'intersection (%)': np.array(v) * 100,
            'k': [k for _ in v]
        })
        for k, v in fit_callback.recommendation_intersection_history.items()
    ])
    fig = px.line(
        df, x="epoch", y="intersection (%)", color='k',
        title='intersection of recommendations on successive epochs'
    )
    if verbose:
        fig.show()
    fig.write_image(workdir / 'recs_intersection.png')


if __name__ == '__main__':
    main(verbose=False)
