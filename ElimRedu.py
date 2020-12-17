import sys
from MassDiff import recommend


def eliminate_redundancy(train_matrix, alpha):
    transition_matrix = (train_matrix.T / train_matrix.T.sum(0)).dot(train_matrix / train_matrix.sum(0))
    transition_matrix += alpha * transition_matrix.dot(transition_matrix)
    initial_resource = train_matrix
    resource = initial_resource.dot(transition_matrix.T) * (1 - train_matrix)
    return resource


if __name__ == "__main__":
    rating_file = sys.path[-1] + "/data/ml100k_data.csv"
    from Preprocessor.load_data import loading

    train_mtx, probe_set = loading(rating_file, 0.9)
    final_res = eliminate_redundancy(train_mtx, -0.75)
    rec_list = recommend(final_res, 50)

    from Evaluation import *

    p = precision(probe_set, rec_list)
    r = recall(probe_set, rec_list, train_mtx.shape[1])
    f1 = harmonic_mean(probe_set, rec_list, train_mtx.shape[1])

    # a_set = np.linspace(-1.5, 0.5, 101)
    # er_p = [precision(probe_set, recommend(eliminate_redundancy(train_mtx, a), 50)) for a in a_set]
    #
    # max_p = max(er_p)
    # alpha_opt = a_set[er_p.index(max_p)]
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.title("ER")
    # plt.xlabel("alpha")
    # plt.ylabel("precision")
    # plt.plot(a_set, er_p, marker="o", c="skyblue")
    # plt.text(alpha_opt+0.2, max_p, "opt=%.2f precision=%.4f" % (alpha_opt, max_p))
    # plt.show()
