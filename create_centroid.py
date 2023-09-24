import numpy as np
import pandas as pd
from pandas import DataFrame
from pathlib import Path


def get_centroid_of_df_cluster(df: DataFrame):
    cnt = len(df.index)
    temp = np.zeros(df.shape[1])
    for i in df.index:
        row = df.loc[i].to_numpy()
        for j in range(len(row)):
            temp[j] += row[j]

    return [x/cnt for x in temp]

if __name__ == "__main__":

    d = {"col1": [5, 2, 7, 1, 3, 4, 7, 8, 2, 4, 5, 1], "col2": [6, 6, 2, 6, 4, 2, 5, 3, 1, 4, 7, 0], "cluster": [1, 0, 1, 1, 0, 1, 2, 2, 2, 0, 2, 1]}
    df = pd.DataFrame(d)

    cluster_criterias = [[] for a in range(3)]
    for x in range(3):
        print(len(df[df['cluster'] == x]))
        cluster_criterias[x] = get_centroid_of_df_cluster(df[df['cluster'] == x])

    print(cluster_criterias)

    saved_cluster = input('enter cluster: ')
    cs = [int(a) for a in saved_cluster.split(',')]
    c = df[df['cluster'].isin(cs)]
    print(c, cs, len(c))
