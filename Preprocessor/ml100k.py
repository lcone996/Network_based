import pandas as pd
import sys


path = "D:/RecSys/dataset/ml-100k/u.data"
org_data = pd.read_csv(path, names=["userId", "itemId", "rating"], sep="\t", usecols=[0, 1, 2])
exp_data = org_data[org_data.rating >= 3]  # 先粗粒化可以剔除孤立用户和物品
del org_data
user_list = list(set(exp_data.userId))
item_list = list(set(exp_data.itemId))

exp_data["userId"] = exp_data["userId"].map(lambda x: user_list.index(x))
exp_data["itemId"] = exp_data["itemId"].map(lambda x: item_list.index(x))

data_info = pd.DataFrame([{"the number of users": len(user_list),
                          "the number of items": len(item_list), "the number of ratings": exp_data.shape[0]}]).T

user_idx_id = pd.DataFrame()
user_idx_id["Id"] = user_list
item_idx_id = pd.DataFrame()
item_idx_id["Id"] = item_list


# 写入预处理后的数据到文件中。数据文件、数据信息、用户物品对应关系
exp_data.to_csv(sys.path[-1]+"/data/ml100k_data.csv", index=False)
data_info.to_csv(sys.path[-1]+"/data/ml100k_info.txt", header=False)
user_idx_id.to_csv(sys.path[-1]+"/data/ml100k_user_id_idx.txt", index=True)
item_idx_id.to_csv(sys.path[-1]+"/data/ml100k_item_id_idx.txt", index=True)
