import pandas as pd
import numpy as np
import scipy


def loading(file_path):
    data = pd.read_csv(file_path)
    group = data.groupby("itemId")
    return
