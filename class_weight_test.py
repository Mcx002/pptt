import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.utils import compute_class_weight

if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[2])
    X = df.to_numpy()

    k_means_optimum = KMeans(n_clusters=int(sys.argv[1]), init='k-means++', n_init='auto', random_state=0)
    y = k_means_optimum.fit_predict(X)

    class_weights = {}
    unique_labels, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    for label, count in zip(unique_labels, counts):
        class_weights[label] = total_samples / (len(unique_labels) * count)

    sample_weights = np.array([class_weights[label] for label in y])

    # classes = np.unique(y)
    # weights = compute_class_weight('balanced', classes=classes, y=y)
    # class_weight = {k: a for k, a in zip(classes, weights)}
    #
    # new_x = []
    # for clus, val in zip(y, X):
    #     row = np.multiply(val, class_weight[clus])
    #     new_x.append(row)

    # new_df = pd.DataFrame(new_x, index=df.index, columns=df.columns)
    k_means_optimum = KMeans(n_clusters=int(sys.argv[1]), init='k-means++', n_init='auto', random_state=0)
    y = k_means_optimum.fit_predict(X, sample_weight=sample_weights)

    df['system_cluster'] = y
    print(y)

    # for i, weight in zip(classes, weights):
    #     df_cluster_prev = df[df['system_cluster'] == i]
    #     df_cluster = df[df['system_cluster'] == i]
    #     for ar in df_cluster.index:
    #         for arc in df_cluster.columns:
    #             if arc == 'system_cluster':
    #                 continue
    #             df_cluster[arc][ar] = df_cluster[arc][ar] * weight
