import numpy as np


def user_hitrate(rank: list, ground_truth: list, k: int = 8):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single hitrate
    """
    return len(set(rank[:k]).intersection(set(ground_truth)))


def user_recall(rank: list, ground_truth: list, k: int = 8):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single recall
    """
    return user_hitrate(rank, ground_truth, k) / len(set(ground_truth))


def user_precision(rank: list, ground_truth: list, k: int = 8):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single precision
    """
    return user_hitrate(rank, ground_truth, k) / len(set(rank[:k]))


def user_ap(rank: list, ground_truth: list, k: int = 8):
    """
    :param rank: shape [n_recommended_items]
    :param ground_truth: shape [n_relevant_items]
    :param k: number of top recommended items
    :return: single ap
    """
    return np.sum([
        user_precision(rank, ground_truth, idx + 1)
        for idx, item in enumerate(rank[:k]) if item in ground_truth
    ]) / len(set(rank[:k]))
