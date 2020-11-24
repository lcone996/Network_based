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

    from Evaluation import precision

    p = precision(probe_set, rec_list)
