import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from Preprocessor.load_data import loading
from MassDiff import recommend, mass_diffusion
from ElimRedu import eliminate_redundancy


def dichotomy(train_matrix, alpha=0, beta=-1, big=True, small=True):
    """
    大度用户消除冗余，小度用户不改变
    :param train_matrix: 训练数据的完整矩阵
    :param alpha: 小度用户参数
    :param beta: 大度用户参数
    :param big: false 跳过大度用户，训练小度用户参数值
    :param small: false 跳过小度用户，训练大度用户参数值
    :return: 资源重分配结果
    """

    # 1.阈值，区分大度用户和小度用户，20%大度用户，80%小度用户
    user_degree = train_matrix.sum(1)
    user_amount, item_amount = train_matrix.shape
    sort_degree = sorted(user_degree)
    threshold = sort_degree[int(user_amount * 0.8)]

    # 2. 划分矩阵
    user = np.arange(user_amount)
    big_user = user[user_degree >= threshold]
    small_user = user[user_degree < threshold]
    big_matrix = train_matrix[big_user, :]
    small_matrix = train_matrix[small_user, :]

    # 3.小度用户
    resource = np.zeros(train_matrix.shape)
    transition_matrix = (train_matrix.T / train_matrix.T.sum(0)).dot(train_matrix / train_matrix.sum(0))
    if small:
        small_transition = transition_matrix + alpha * transition_matrix.dot(transition_matrix)
        small_resource = small_matrix.dot(small_transition.T) * (1 - small_matrix)
        resource[small_user, :] = small_resource
        if not big:
            return resource, small_user

    # 4.大度用户
    if big:
        big_transition = transition_matrix + beta * transition_matrix.dot(transition_matrix)
        big_resource = big_matrix.dot(big_transition.T) * (1 - big_matrix)
        resource[big_user, :] = big_resource
        if not small:
            return resource, big_user

    return resource


if __name__ == "__main__":
    rating_file = sys.path[-1] + "/data/ml100k_data.csv"
    train_mtx, probe_set = loading(rating_file, 0.9)

    md_res = mass_diffusion(train_mtx)
    md_rec = recommend(md_res, 50)

    er_res = eliminate_redundancy(train_mtx, -0.75)
    er_rec = recommend(er_res, 50)

    ex_res = dichotomy(train_mtx, alpha=0, beta=-1)
    ex_rec = recommend(ex_res, 50)

    ex_res_para = dichotomy(train_mtx, alpha=-0.9, beta=-0.87)
    ex_rec_para = recommend(ex_res_para, 50)
    from Evaluation import *

    p = precision(probe_set, ex_rec_para)
    r = recall(probe_set, ex_rec_para, train_mtx.shape[1])
    f1 = harmonic_mean(probe_set, ex_rec_para, train_mtx.shape[1])
    h = hamming_distance(ex_rec_para)
    a = average_degree(train_mtx.sum(0), ex_rec_para)

    md_p = precision(probe_set, md_rec)
    er_p = precision(probe_set, er_rec)
    ex_p = precision(probe_set, ex_rec)
    ex_p_para = precision(probe_set, ex_rec_para)

    # 训练小度用户参数值
    # a_range = np.linspace(-1.5, 0, 151)
    # ex_p = []
    # for a in a_range:
    #     small_res, small_user = dichotomy(train_mtx, alpha=a, big=False)
    #     small_probe = probe_set[probe_set["userId"].isin(small_user)]
    #     p = precision(small_probe, recommend(small_res, 50))/len(small_user)*small_res.shape[0]
    #     ex_p.append(p)
    #     print("alpha=%.2f, precision=%.4f" % (a, p))
    #
    # a_opt = a_range[ex_p.index(max(ex_p))]
    # plt.figure()
    # plt.title("parameter of inactive users")
    # plt.xlabel("alpha")
    # plt.ylabel("precision")
    # plt.plot(a_range, ex_p, marker="o", c="r")
    # plt.text(a_opt+0.2, max(ex_p), "opt=%.2f precision=%.4f" % (a_opt, max(ex_p)))
    # plt.show()

    # 训练大度用户参数值
    # b_range = np.linspace(-1.5, 0, 151)
    # ex_p = []
    # for b in b_range:
    #     big_res, big_user = dichotomy(train_mtx, beta=b, small=False)
    #     big_probe = probe_set[probe_set["userId"].isin(big_user)]
    #     p = precision(big_probe, recommend(big_res, 50)) / len(big_user) * big_res.shape[0]
    #     ex_p.append(p)
    #     print("beta=%.2f, precision=%.4f" % (b, p))
    #
    # b_opt = b_range[ex_p.index(max(ex_p))]
    # plt.figure()
    # plt.title("parameter of active users")
    # plt.xlabel("beta")
    # plt.ylabel("precision")
    # plt.plot(b_range, ex_p, marker="o", c="r")
    # plt.text(b_opt + 0.2, max(ex_p), "opt=%.2f, precision=%.4f" % (b_opt, max(ex_p)))
    # plt.show()
