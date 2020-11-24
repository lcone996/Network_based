import pandas as pd
import numpy as np
import scipy.sparse as ss
import sys


def loading(file_path, alpha):
    data = pd.read_csv(file_path)
    train = [x[1].sample(frac=alpha) for x in data.groupby("itemId")]  # groupby (itemId, dataframe)
    train = pd.concat(train)
    probe = data[~data.isin(train)].dropna()

    value = np.ones(train.shape[0])
    item = train["itemId"]
    user = train["userId"]
    user_amount = len(set(user))
    item_amount = len(set(item))
    matrix = ss.csc_matrix((value, (user, item)), shape=[user_amount, item_amount]).toarray()

    return matrix, probe


if __name__ == "__main__":
    rating_file = sys.path[-1] + "/data/ml100k_data.csv"
    train_mtx, probe_set = loading(rating_file, 0.9)
