import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from jqmcvi.base import dunn_fast
from pandas import DataFrame
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from utils import text_preprocessing, silhouette_checking, elbow_method, kmeans, beep, get_centroid_of_df_cluster

# List of public figures for reference
public_figure = ['anies', 'ganjar', 'prabowo', 'puan', 'ridwan', 'ahok', 'kamil', 'ridw']

class AutomateRemodelDataset:
    def __init__(
            self,
            *,
            pickle_path='',
            min_iter=2,
            max_iter=18,
            algorithm='kmeans',
            min_eps=35,
            max_eps=101,
            jump_eps=15,
    ):
        """
        Initialize the AutomateRemodelDataset class.
        """
        # Initialize class variables with provided parameters
        self.folder_path = prepare_path()
        self.pickle_path = pickle_path
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.iter_num = 1
        self.algorithm = algorithm
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.jump_eps = jump_eps

    def load_data(self):
        data, tf_idfs = text_preprocessing()
        df = pd.DataFrame(tf_idfs).fillna(0)

        file_name = time.time()
        self.pickle_path = '{}/{}.pkl.xz'.format(self.folder_path, file_name)
        df.to_pickle(self.pickle_path)

    def load_pickle(self):
        return pd.read_pickle(self.pickle_path)

    def find_best_eval_kmeans(self, x):
        """
       Find the optimal number of clusters using the K-Means algorithm.

       Parameters:
       - x: array-like
           Input data for clustering.

       Returns:
       - n_cluster: int
           The optimal number of clusters.
       """
        # Find the optimal number of clusters using K-Means
        # Prepare forecast logs
        silhouette_file_path = '{}/{}_elbow_best_silhouette.md'.format(self.folder_path, time.time())
        f = open(silhouette_file_path, 'a')
        f.write('# Search Elbow and Best Silhouette score\n')
        f.write('\n')

        # # Generate Elbow Forecast
        # elbow_file_name = '{}_elbow.png'.format(time.time())
        # elbow_file_path = '{}/{}'.format(self.folder_path, elbow_file_name)
        # elbow_knee = elbow_method(x, min_iter=self.min_iter, max_iter=self.max_iter, save_image=elbow_file_path)
        #
        # # Write Elbow Knee
        # f.write('Elbow Knee = {}\n'.format(elbow_knee))
        f.close()

        # Create silhouette forecasting
        n_cluster, silhouette_logs = silhouette_checking(x, min_iter=self.min_iter, max_iter=self.max_iter,
                                                         show_log=True)

        # Write silhouette table
        f = open(silhouette_file_path, 'a')
        f.write('Best Silhouette Score = {}\n'.format(n_cluster))
        f.write('\n')
        f.write('| Cluster | Silhouette Score |\n')
        f.write('| --- | --- |\n')
        for i in range(len(silhouette_logs)):
            f.write('| {} | {} |\n'.format(silhouette_logs[i]['cluster'], silhouette_logs[i]['silhouette_score']))

        f.close()
        return n_cluster

    def k_means(self, df: DataFrame, X: [], n_cluster: int):
        """
        Perform K-Means clustering on the input data and save the results.

        Parameters:
        - df: DataFrame
            Input data as a DataFrame.
        - X: array-like
            Input data for clustering.
        - n_cluster: int
            Number of clusters to create.
        """
        # Perform K-Means clustering and save results
        y, _, silhouette, dunn_index, df_percentage = kmeans(df, X, n_cluster)
        df['system_cluster'] = y

        # Writing the log
        file_path = '{}/{}_k_means.md'.format(self.folder_path, time.time())
        f = open(file_path, 'a')
        f.write('# K-Means Cluster {}\n'.format(n_cluster))
        f.write('\n')
        f.write('Total Article Amount: {}\n'.format(len(y)))
        f.write('Silhouette Score {}\n'.format(silhouette))
        f.write('Dunn Index {}\n'.format(dunn_index))
        f.write('\n')
        f.write('-----\n')

        cs = []

        cluster_sizes = []

        for i in range(n_cluster):
            size = len(y[y == i])
            cluster_sizes.append(size)

            f.write('cluster {}\n'.format(i + 1))
            f.write('size {}\n'.format(size))
            f.write('\n')
            f.write('| word | score |\n')
            f.write('| --- | --- |\n')
            d = df_percentage.T[i].sort_values(ascending=0)
            for j in range(20):
                f.write('| {} | {} |\n'.format(d.index[j], d[j]))

                if j <= 5 and d.index[j] in public_figure:
                    cs.append(i)

            f.write('\n')
            f.write('\n')
            f.write('-----\n')
            f.write('\n')
            f.write('\n')
        cs = np.unique(cs)
        f.close()

        cluster_sizes.sort()
        sum_cluster_sizes = np.sum(cluster_sizes)/len(cluster_sizes)
        if abs(cluster_sizes[0]-sum_cluster_sizes) < 100 and abs(cluster_sizes[-1]-sum_cluster_sizes) < 100:
            raise StopIteration()

        if len(cs) == n_cluster:
            del df['system_cluster']
            next_n_cluster = n_cluster + 1
            self.k_means(df, X, next_n_cluster)
        else:
            self.pickle_path = '{}/{}.pkl.xz'.format(self.folder_path, time.time())
            df_cs = df[df['system_cluster'].isin(cs)]
            del df_cs['system_cluster']
            df_cs.to_pickle(self.pickle_path)

        # if len(cs) == n_cluster:
        #     cs_cluster = -1
        #     cs_size = -1
        #     for k in cs:
        #         size = len(y[y == k])
        #         if size > cs_size:
        #             cs_size = size
        #             cs_cluster = k
        #
        #     cs = [cs_cluster]

    def find_best_epsilon(self):
        """
        Find the best epsilon value for DBSCAN clustering and save the results.

        Returns:
        - best_eps: int
            The best epsilon value found.
        - info: DataFrame
            Information about clustering results for different epsilon values.
        """
        rnge = range(self.min_eps, self.max_eps, self.jump_eps)

        # load data from save file
        df = pd.read_pickle(pickle_path)
        X = df.to_numpy()

        # write header
        file_name = time.time()
        f = open('{}/{}_find_best_epsilon.md'.format(self.folder_path, file_name), 'a')
        f.write('# Find Best Epsilon')
        f.write('\n')
        f.write('| Epsilon | Cluster | Silhoette Score | Dunn Index |\n')
        f.write('| --- | --- | --- | --- |\n')
        f.close()

        best_silhoette_avg = -1
        best_eps = -1
        epss = []
        clusters = []
        silhouettes = []
        dunns = []

        for i in rnge:
            # i = 65
            clustering = DBSCAN(eps=i, min_samples=4).fit(X)
            y = clustering.labels_

            silhouette_avg = -1
            try:
                silhouette_avg = silhouette_score(X, y)
            except ValueError:
                print('silhouette error')

            dunn_avg = dunn_fast(X, y)
            n_cluster = len(set(y)) - (1 if -1 in y else 0)
            f = open('{}/{}_find_best_epsilon.md'.format(self.folder_path, file_name), 'a')
            f.write('| {} | {} | {} | {} |\n'.format(i, n_cluster, silhouette_avg, dunn_avg))
            f.close()

            epss.append(i)
            clusters.append(n_cluster)
            silhouettes.append(silhouette_avg)
            dunns.append(dunn_avg)

            if silhouette_avg > best_silhoette_avg and n_cluster != 1:
                best_silhoette_avg = silhouette_avg
                best_eps = i

        info = pd.DataFrame([clusters, silhouettes, dunns], index=['cluster', 'silhouette', 'dunn index'], columns=epss)
        return best_eps, info.T

    def dbscan_clustering(self, df, x, eps, info):
        """
        Perform DBSCAN clustering on the input data and save the results.

        Parameters:
        - df: DataFrame
            Input data as a DataFrame.
        - x: array-like
            Input data for clustering.
        - eps: int
            Epsilon value for DBSCAN clustering.
        - info: DataFrame
            Information about clustering results for different epsilon values.
        """
        # Perform DBSCAN clustering and save results
        clustering = DBSCAN(eps=eps, min_samples=4).fit(x)
        y = clustering.labels_
        cluster = len([a for a in np.unique(y) if a != -1])

        # Evaluate cluster
        silhouette = -1
        if cluster > 1:
            silhouette = silhouette_score(x, y)
        dunn_index = dunn_fast(x, y)

        df['system_cluster'] = y

        cluster_criterias = [[] for _ in range(eps)]
        for x in range(cluster):
            cluster_criterias[x] = get_centroid_of_df_cluster(df[df['system_cluster'] == x])
        centroids = pd.DataFrame(cluster_criterias, columns=df.columns)

        # Writing the log
        file_name = time.time()
        f = open('{}/{}_dbscan_clustering.md'.format(self.folder_path, file_name), 'a')

        f.write('# DBSCAN Cluster {}\n'.format(cluster))
        f.write('Size {}\n'.format(len(y)))
        f.write('Silhouette Score {}\n'.format(silhouette))
        f.write('Dunn Index {}\n'.format(dunn_index))
        f.write('\n')
        f.write('-----\n')

        cluster_sizes = []
        cs = []

        for i in range(cluster):
            size = len(df[df['cluster'] == i])
            cluster_sizes.append(size)
            f.write('cluster {}\n'.format(i + 1))
            f.write('size {}\n'.format(size))
            f.write('\n')
            f.write('| word | score |\n')

            d = centroids.T[i].sort_values(ascending=0)
            for j in range(20):
                f.write('| {} | {} |\n'.format(d.index[j], d[j]))

                if j <= 5 and d.index[j] in public_figure:
                    cs.append(i)
        f.write('\n')
        f.write('\n')
        f.write('-----\n')
        f.write('\n')
        f.write('\n')
        f.close()

        cluster_sizes.sort()
        sum_cluster_sizes = np.sum(cluster_sizes)/len(cluster_sizes)
        if abs(cluster_sizes[0]-sum_cluster_sizes) < 100 and abs(cluster_sizes[-1]-sum_cluster_sizes) < 100:
            raise StopIteration()

        if len(cs) == cluster:
            del df['system_cluster']

            next_n_cluster = cluster + 1
            is_cluster_undefined = next_n_cluster not in info['cluster'].to_numpy()
            while is_cluster_undefined:
                next_n_cluster = next_n_cluster + 1
                is_cluster_undefined = next_n_cluster not in info['cluster'].to_numpy()

            new_eps = info[info['cluster'] == next_n_cluster].tail(1).index[0]

            self.dbscan_clustering(df, x, new_eps, info)
        else:
            self.pickle_path = '{}/{}.pkl.xz'.format(self.folder_path, time.time())
            df_cs = df[df['system_cluster'].isin(cs)]
            del df_cs['system_cluster']
            df_cs.to_pickle(self.pickle_path)

    def automate(self):
        """
        Automate the clustering process based on the specified algorithm.
        """
        # Automate the clustering process
        print('---------------------------------------')
        print()
        print('Iteration number:', self.iter_num)
        if self.pickle_path == "":
            print('pickle path not found')
            self.load_data()

        df = self.load_pickle()
        X = df.to_numpy()
        try:
            if self.algorithm == 'kmeans':
                n_cluster = self.find_best_eval_kmeans(X)
                self.k_means(df, X, n_cluster)
            if self.algorithm == 'dbscan':
                n_eps, info = self.find_best_epsilon()
                self.dbscan_clustering(df, X, n_eps, info)
        except StopIteration:
            return
        except ValueError as err:
            print(err)
            beep()
            return

        self.iter_num = self.iter_num + 1
        print()
        return self.automate()


def prepare_path():
    """
    Prepare and return a folder path for saving clustering results.

    Returns:
    - folder_path: str
        Folder path for saving results.
    """
    # Prepare and return a folder path for saving clustering results
    # Prepare Path
    folder_name = time.time()
    folder_path = './data/automate/{}'.format(folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print('please input proper arguments')
    #     sys.exit()

    pickle_path = ""
    if len(sys.argv) > 2:
        pickle_path = sys.argv[2]

    ard = AutomateRemodelDataset(algorithm=sys.argv[1], pickle_path=pickle_path)
    ard.automate()

    print('Program finished')