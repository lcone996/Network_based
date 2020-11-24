import sys


def mass_diffusion(train_matrix):
    transition_matrix = (train_matrix.T / train_matrix.T.sum(0)).dot(train_matrix / train_matrix.sum(0))
    initial_resource = train_matrix
    resource = initial_resource.dot(transition_matrix.T) * (1 - train_matrix)
    return resource


def recommend(final_resource, length):
    user_amount = final_resource.shape[0]
    recommend_list = [[]] * user_amount
    for i in range(user_amount):
        temp = sorted(enumerate(final_resource[i, :]), key=lambda x: x[1], reverse=True)
        recommend_list[i] = [x[0] for x in temp[:length]]
    return recommend_list


if __name__ == "__main__":
    rating_file = sys.path[-1] + "/data/ml100k_data.csv"
    from Preprocessor.load_data import loading

    train_mtx, probe_set = loading(rating_file, 0.9)
    final_res = mass_diffusion(train_mtx)
    rec_list = recommend(final_res, 50)

    from Evaluation import precision

    p = precision(probe_set, rec_list)
