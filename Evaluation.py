import numpy as np
import pandas as pd


def precision(probe_set, recommend_list):
    group = probe_set.groupby("userId")
    count = [len(set(recommend_list[int(x[0])]) & set(x[1]["itemId"])) for x in group]
    return np.sum(count) / len(recommend_list[0]) / len(recommend_list)


def recall(probe_set, recommend_list, item_amount):
    probe_count = probe_set["itemId"].value_counts()
    hit_count = np.zeros(item_amount)
    for cell in probe_set.itertuples(index=False):
        user = int(cell[0])
        item = int(cell[1])
        if item in recommend_list[user]:
            hit_count[item] += 1
    rec = [hit_count[int(item)] / probe_count[item] for item in probe_count.index]
    return np.mean(rec)


def harmonic_mean(probe_set, recommend_list, item_amount):
    p = precision(probe_set, recommend_list)
    r = recall(probe_set, recommend_list, item_amount)
    return 2 * p * r / (p + r)


def hamming_distance(recommend_list):
    """
    不同用户之间推荐列表的差异性
    """
    d = 0
    user_amount = len(recommend_list)
    for i in range(user_amount - 1):
        for j in range(i + 1, user_amount):
            d += len(set(recommend_list[i]) & set(recommend_list[j]))
    return 1 - 2 * d / user_amount / (user_amount - 1) / len(recommend_list[0])


def average_degree(item_degree, recommend_list):
    d = 0
    for rec in recommend_list:
        d += sum(item_degree[rec])
    return round(d / len(recommend_list) / len(recommend_list[0]))


def all_measure(rec_list, probe_set, train_mtx):
    p = precision(probe_set, rec_list)
    # r = recall(probe_set, rec_list, train_mtx.shape[1])
    # f1 = harmonic_mean(probe_set, rec_list, train_mtx.shape[1])
    h = hamming_distance(rec_list)
    a = average_degree(train_mtx.sum(0), rec_list)
    print("precision = %.4f" % p)
    print("hamming = %.4f" % h)
    print("novelty = %d" % a)
