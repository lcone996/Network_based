import numpy as np


def precision(probe_set, recommend_list):
    group = probe_set.groupby("userId")
    count = [len(set(recommend_list[int(x[0])]) & set(x[1]["itemId"])) for x in group]
    return np.sum(count) / len(recommend_list[0]) / len(recommend_list)
