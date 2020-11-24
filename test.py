import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from Preprocessor.load_data import loading
from MassDiff import mass_diffusion, recommend
from ElimRedu import eliminate_redundancy

rating_file = sys.path[-1] + "/data/ml100k_data.csv"
train_mtx, probe_set = loading(rating_file, 0.9)

user_amount, item_amount = train_mtx.shape
user_degree = train_mtx.sum(1)
item_degree = train_mtx.sum(0)

length = 50
md_list = recommend(mass_diffusion(train_mtx), 50)
er_list = recommend(eliminate_redundancy(train_mtx, -0.75), 50)

md_pre, er_pre = [0] * user_amount, [0] * user_amount
group = probe_set.groupby("userId")
for x in group:
    user = int(x[0])
    rec = set(x[1]["itemId"])
    md_pre[user] = len(rec & set(md_list[user])) / length
    er_pre[user] = len(rec & set(er_list[user])) / length

df = pd.DataFrame()
df["userId"] = np.arange(user_amount)
df["user_degree"] = user_degree
df["md_pre"] = md_pre
df["er_pre"] = er_pre
df.sort_values("user_degree", ascending=True, inplace=True)

plt.figure()
plt.plot(df["user_degree"], df["md_pre"])
plt.plot(df["user_degree"], df["er_pre"])
plt.show()

plt.figure()
plt.plot(df["user_degree"], df["er_pre"] - df["md_pre"])
plt.show()
