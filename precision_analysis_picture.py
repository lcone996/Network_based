import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from Preprocessor.load_data import loading
from MassDiff import mass_diffusion, recommend
from ElimRedu import eliminate_redundancy
from explor import dichotomy


def precision_analysis(rec_list1, rec_list2, train_matrix, probe_set, *image_title):
    list_length = len(rec_list1[0])
    user_amount = train_matrix.shape[0]
    user_degree = train_matrix.sum(1)
    precision1, precision2 = np.zeros(user_amount), np.zeros(user_amount)
    for x in probe_set.groupby("userId"):
        user = int(x[0])
        items = set(x[1]["itemId"])
        precision1[user] = len(items & set(rec_list1[user]))
        precision2[user] = len(items & set(rec_list2[user]))
    precision1 /= list_length
    precision2 /= list_length

    err_df = pd.DataFrame()
    err_df["user_degree"] = user_degree
    err_df["error"] = precision2 - precision1
    err_df["count"] = np.ones(user_amount)
    group = err_df.groupby(["user_degree", "error"]).count()

    count_df = pd.DataFrame()
    count_df["degree"] = [x[0] for x in group.index.values]
    count_df["error"] = [x[1] for x in group.index.values]
    count_df["count"] = list(group["count"])

    plt.figure()
    plt.title(image_title[0])
    plt.xlabel("user degree")
    plt.ylabel("error of precision")
    plt.ylim(-0.15, 0.3)
    pos_sample = count_df[count_df["error"] > 0]
    neg_sample = count_df[count_df["error"] < 0]
    not_sample = count_df[count_df["error"] == 0]
    s1 = plt.scatter(pos_sample["degree"], pos_sample["error"], c="y", marker="^", s=pos_sample["count"] * 3 + 7)
    s2 = plt.scatter(neg_sample["degree"], neg_sample["error"], c="r", marker="v", s=neg_sample["count"] * 3 + 7)
    s3 = plt.scatter(not_sample["degree"], not_sample["error"], c="skyblue", marker="o",
                     s=not_sample["count"] * 3 + 7)
    fra1 = "%.1f%%" % (pos_sample["count"].sum() / user_amount * 100)
    fra2 = "%.1f%%" % (neg_sample["count"].sum() / user_amount * 100)
    fra3 = "%.1f%%" % (not_sample["count"].sum() / user_amount * 100)
    plt.legend(handles=[s1, s2, s3], labels=["positive", "negative", "unchanged"])
    plt.text(400, 0.1, fra1)
    plt.text(400, 0, fra3)
    plt.text(400, -0.05, fra2)
    plt.show()


if __name__ == "__main__":
    rating_file = sys.path[-1] + "/data/ml100k_data.csv"
    train_mtx, probe = loading(rating_file, 0.9)

    rec1 = recommend(mass_diffusion(train_mtx), 50)
    rec2 = recommend(eliminate_redundancy(train_mtx, -0.75), 50)
    rec3 = recommend(dichotomy(train_mtx, alpha=-0.9, beta=-0.87), 50)

    precision_analysis(rec1, rec2, train_mtx, probe, "MD & ER")
    precision_analysis(rec1, rec3, train_mtx, probe, "MD & ER-2")
