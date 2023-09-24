import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from progress.bar import Bar

from utils import (
    kmeans,
    elbow_method,
    silhouette_checking,
    get_centroid_of_df_cluster,
    text_preprocessing, beep, plot_dendrogram
)
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from jqmcvi.base import dunn_fast
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import RandomOverSampler

pickle = '/home/damian/research/python/preprocessing-text/data/important/detik_weightened.pkl.xz'


def k_means(save, clusters):
    clusters = np.unique(clusters)
    bar = Bar('Process', max=2+(2*len(clusters)))
    # # load data from save file
    df = pd.read_pickle(pickle)
    bar.next()

    X = df.to_numpy()
    bar.next()
    file_name = time.time()
    for cluster in clusters:
        y, _, silhouette, dunn_index, df_percentage = kmeans(df, X, cluster)
        df['system_cluster'] = y
        bar.next()

        # Writing the log
        folder_path = './data/log'
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        f = open('{}/{}_k_means.md'.format(folder_path, file_name), 'a')
        f.write('# K-Means Cluster {}\n'.format(cluster))
        f.write('Total Article Amount: {}\n'.format(len(y)))
        f.write('Silhouette Score {}\n'.format(silhouette))
        f.write('Dunn Index {}\n'.format(dunn_index))
        f.write('\n')
        f.write('-----\n')

        for i in range(cluster):
            f.write('cluster {}\n'.format(i+1))
            f.write('size {}\n'.format(len(y[y == i])))
            f.write('\n')
            f.write('| word | score |\n')
            f.write('| --- | --- |\n')
            d = df_percentage.T[i].sort_values(ascending=0)
            for j in range(80):
                f.write('| {} | {} |\n'.format(d.index[j], d[j]))
            f.write('\n')
            f.write('\n')
            f.write('-----\n')
        f.write('\n')
        f.write('\n')
        f.close()

        if 'system_cluster' in df:
            del df['system_cluster']

        beep()

        if save:

            print('If you want to put 2 cluster or more, divide the cluster by ","')
            saved_cluster = input("Enter Clusters will be saved: ")
            cs = [int(a)-1 for a in saved_cluster.split(',')]
            print('cluster index filter:', cs)

            df_cs = df[df['system_cluster'].isin(cs)]
            print('prev amount is {} and saved amount will be {}'.format(len(y), len(df_cs)))
            sure = input('are you sure to save the data? [0|1]')
            sure = int(sure)

            if sure:

                folder_path = './data/saved'
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                file_name = time.time()
                file_path = '{}/{}.pkl.xz'.format(folder_path, file_name)
                del df_cs['system_cluster']
                df_cs.to_pickle(file_path)
                print('file {}.pkl.xz has been saved'.format(file_name))

        bar.next()

    bar.finish()


def search_elbow_and_best_silhouette(min_iter, max_iter, show_graphic):
    bar = Bar('Process', max=2)
    # load data from save file
    df = pd.read_pickle(pickle)
    bar.next()

    X = df.to_numpy()
    bar.next()

    file_name = time.time()
    folder_path = './data/log'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    f = open('{}/{}_elbow_best_silhouette.md'.format(folder_path, file_name), 'a')
    f.write('# Search Elbow and Best Silhouette score\n')
    f.write('\n')
    f.write('\n')

    elbow_img_path = '{}/{}_elbow.png'.format('/home/damian/research/python/preprocessing-text/data/log', time.time())
    elbow_knee = elbow_method(X, min_iter=min_iter, max_iter=max_iter, save_image=elbow_img_path)
    f.write('Elbow Knee = {}\n'.format(elbow_knee))
    f.close()

    f = open('{}/{}_elbow_best_silhouette.md'.format(folder_path, file_name), 'a')
    n_cluster, silhouette_logs = silhouette_checking(X, min_iter=min_iter, max_iter=max_iter, show_log=True)
    f.write('Best Silhouette Score = {}\n'.format(n_cluster))

    # Silhouette Logs
    f.write('| Cluster | Silhouette Score |\n')
    f.write('| --- | --- |\n')
    for i in range(len(silhouette_logs)):
        f.write('| {} | {} |\n'.format(silhouette_logs[i]['cluster'], silhouette_logs[i]['silhouette_score']))

    f.close()


def best_agglomerative_cluster(min_iter, max_iter):
    bar = Bar('Process', max=2+(max_iter - min_iter))
    # load data from save file
    df = pd.read_pickle(pickle)
    bar.next()

    X = df.to_numpy()
    bar.next()
    file_name = time.time()
    folder_path = './data/log'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    f = open('{}/{}_best_agglomemrative_cluster.md'.format(folder_path, file_name), 'a')
    f.write('# Best Agglomerative Clustering\n')
    f.write('\n')
    f.write('| Cluster | Silhouette Score | Dunn Index |\n')
    f.write('| --- | --- | --- |\n')
    f.close()
    for i in range(min_iter, max_iter):
        f = open('{}/{}_best_agglomemrative_cluster.md'.format(folder_path, file_name), 'a')
        clustering = AgglomerativeClustering(n_clusters=i).fit(X)
        # Evaluate cluster
        silhouette_avg = silhouette_score(X, clustering.labels_)
        dunn_avg = dunn_fast(X, clustering.labels_)
        f.write('| {} | {} | {} |\n'.format(i, silhouette_avg, dunn_avg))
        f.close()
        print(' >  {}  {}  {} \n'.format(i, silhouette_avg, dunn_avg))
        bar.next()

    bar.finish()


def find_nearest_neighbors_graphic():
    bar = Bar('Process', max=2)
    # load data from save file
    df = pd.read_pickle(pickle)
    bar.next()

    X = df.to_numpy()
    neighbors = NearestNeighbors(n_neighbors=2).fit(X)
    bar.next()
    bar.finish()
    distances = neighbors.kneighbors(X)
    distances = np.sort(distances[0], axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.xlabel('Points (sample) sorted by distance')
    plt.ylabel('Epsilon')
    # plt.axis([2800, 4250, 30, 225])
    plt.show()


def find_best_epsilon(min_eps, max_eps, jump_eps):
    rnge = range(min_eps, max_eps, jump_eps)
    bar = Bar('Process', max=2+(len(rnge)))
    # load data from save file
    df = pd.read_pickle(pickle)
    bar.next()

    X = df.to_numpy()
    file_name = time.time()
    folder_path = './data/log'
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # write header
    f = open('{}/{}_find_best_epsilon.md'.format(folder_path, file_name), 'a')
    f.write('# Find Best Epsilon')
    f.write('\n')
    f.write('| Epsilon | Cluster | Silhoette Score | Dunn Index |\n')
    f.write('| --- | --- | --- | --- |\n')
    f.close()

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
        f = open('{}/{}_find_best_epsilon.md'.format(folder_path, file_name), 'a')
        f.write('| {} | {} | {} | {} |\n'.format(i, len(set(y)) - (1 if -1 in y else 0), silhouette_avg, dunn_avg))
        f.close()

        bar.next()

    bar.finish()


def agglomerative_clustering(save, cluster):
    bar = Bar('Process', max=6)
    df = pd.read_pickle(pickle)
    X = df.to_numpy()
    bar.next()
    clustering = AgglomerativeClustering(n_clusters=cluster).fit(X)
    y = clustering.labels_
    bar.next()
    # Evaluate cluster
    silhouette = silhouette_score(X, y)
    bar.next()
    dunn_index = dunn_fast(X, y)
    bar.next()
    df['system_cluster'] = y
    cluster_criterias = [[] for a in range(cluster)]
    for x in range(cluster):
        cluster_criterias[x] = get_centroid_of_df_cluster(df[df['system_cluster'] == x])
    centroids = pd.DataFrame(cluster_criterias, columns=df.columns)
    # Writing the log
    file_name = time.time()

    bar.next()
    folder_path = './data/log'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    f = open('{}/{}_agglomerative_clustering.md'.format(folder_path, file_name), 'a')
    f.write('# Agglomerative Cluster {}\n'.format(cluster))
    f.write('\n')
    f.write('Size {}\n'.format(len(clustering.labels_)))
    f.write('Silhouette Score {}\n'.format(silhouette))
    f.write('Dunn Index {}\n'.format(dunn_index))
    f.write('\n')
    f.write('-----\n')

    for i in range(cluster):
        f.write('cluster {}\n'.format(i+1))
        f.write('size {}\n'.format(len(df[df['system_cluster'] == i])))
        f.write('\n')
        f.write('| word | score |\n')
        f.write('| --- | --- |\n')

        d = centroids.T[i].sort_values(ascending=0)
        for j in range(80):
            f.write('| {} | {} |\n'.format(d.index[j], d[j]))

    f.write('\n')
    f.write('\n')
    f.write('-----\n')
    f.write('\n')
    f.write('\n')
    f.close()

    if save:
        beep()
        print('If you want to put 2 cluster or more, divide the cluster by ","')
        saved_cluster = input("Enter Clusters will be saved: ")
        cs = [int(a)-1 for a in saved_cluster.split(',')]
        folder_path = './data/saved'
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_name = time.time()
        df[df['system_cluster'].isin(cs)].to_pickle('{}/{}.pkl.xz'.format(folder_path, file_name))

    bar.next()
    bar.finish()


def dbscan_clustering(save, epsilon):
    bar = Bar('Process', max=6)
    df = pd.read_pickle(pickle)
    X = df.to_numpy()
    bar.next()
    clustering = DBSCAN(eps=epsilon, min_samples=4).fit(X)
    bar.next()
    # Evaluate cluster
    silhouette = silhouette_score(X, clustering.labels_)
    bar.next()
    dunn_index = dunn_fast(X, clustering.labels_)
    bar.next()
    df['system_cluster'] = clustering.labels_
    cluster = len([a for a in np.unique(clustering.labels_) if a != -1])

    cluster_criterias = [[] for a in range(epsilon)]
    for x in range(cluster):
        cluster_criterias[x] = get_centroid_of_df_cluster(df[df['system_cluster'] == x])
    centroids = pd.DataFrame(cluster_criterias, columns=df.columns)
    bar.next()

    # Writing the log
    folder_path = './data/log'
    file_name = time.time()
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    f = open('{}/{}_dbscan_clustering.md'.format(folder_path, file_name), 'a')

    f.write('# DBSCAN Cluster {}\n'.format(cluster))
    f.write('\n')
    f.write('Size {}\n'.format(len(clustering.labels_)))
    f.write('Silhouette Score {}\n'.format(silhouette))
    f.write('Dunn Index {}\n'.format(dunn_index))
    f.write('\n')
    f.write('-----\n')

    for i in range(cluster):
        f.write('cluster {}\n'.format(i+1))
        f.write('size {}\n'.format(len(df[df['system_cluster'] == i])))
        f.write('\n')
        f.write('| word | score |\n')

        d = centroids.T[i].sort_values(ascending=0)
        for j in range(80):
            f.write('| {} | {} |\n'.format(d.index[j], d[j]))
    f.write('\n')
    f.write('\n')
    f.write('-----\n')
    f.write('\n')
    f.write('\n')
    f.close()

    if save:
        beep()
        print('If you want to put 2 cluster or more, divide the cluster by ","')
        saved_cluster = input("Enter Clusters will be saved: ")
        cs = [int(a)-1 for a in saved_cluster.split(',')]

        df_cs = df[df['system_cluster'].isin(cs)]
        print('prev amount is {} and saved amount will be {}'.format(len(clustering.labels_), len(df_cs)))
        sure = input('are you sure to save the data? [0|1]')
        sure = int(sure)

        if sure:
            folder_path = './data/saved'
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_name = time.time()
            df_cs = df[df['system_cluster'].isin(cs)]
            del df_cs['system_cluster']
            df_cs.to_pickle('{}/{}.pkl.xz'.format(folder_path, file_name))

            print('file {}.pkl.xz has been saved'.format(file_name))

    bar.next()
    bar.finish()


def agglomerative_dendrogram():
    df = pd.read_pickle(pickle)
    X = df.to_numpy()

    model = AgglomerativeClustering(n_clusters=2, linkage='ward')
    model = model.fit(X)

    y = model.labels_
    for val in np.unique(y):
        print('cluster {} size is {}'.format(val + 1, len(y[y==val])))

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')

    model = model.fit(X)
    plot_dendrogram(model, truncate_mode="level", p=40)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    print(model)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please put the valid argument')
        sys.exit()

    is_help = sys.argv[1] == '-h'

    option = sys.argv[1]

    if is_help:
        print('kmeans           <save:bool> <...cluster>')
    if option == 'kmeans':
        if len(sys.argv) < 4:
            print('k-means: doesnt have proper args')
            sys.exit()
        k_means(int(sys.argv[2]), [int(a) for a in sys.argv[3:]])

    if is_help:
        print('elbow            <min_iter:int> <max_iter:int> <show_graphic:bool>')
    if option == 'elbow':
        if len(sys.argv) < 5:
            print('search-elbow: doesnt have proper args')
            sys.exit()
        search_elbow_and_best_silhouette(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    if is_help:
        print('bestaggr         <min_iter> <max_iter>')
    if option == 'bestaggr':
        if len(sys.argv) < 4:
            print('best-agglomerative-cluster: doesnt have proper args')
            sys.exit()
        best_agglomerative_cluster(int(sys.argv[2]), int(sys.argv[3]))

    if is_help:
        print('knn')
    if option == 'knn':
        print('hi there')
        find_nearest_neighbors_graphic()

    if is_help:
        print('besteps          <min_eps:int> <max_eps:int> <jump_eps:int>')
    if option == 'besteps':
        find_best_epsilon(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    if is_help:
        print('agglomerative    <save:boolean> <cluster:int>')
    if option == 'agglomerative':
        agglomerative_clustering(int(sys.argv[2]), int(sys.argv[3]))

    if is_help:
        print('dbscan           <save:boolean> <epsilon:int>')
    if option == 'dbscan':
        dbscan_clustering(int(sys.argv[2]), int(sys.argv[3]))

    if option == 'n':
        data, tf_idfs = text_preprocessing()
        df = pd.DataFrame(tf_idfs).fillna(0)

        folder_path = './data/saved'
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_name = time.time()
        df.to_pickle('{}/{}.pkl.xz'.format(folder_path, file_name))

    if option == 'aggdendogram':
        agglomerative_dendrogram()

    beep()
    # centroids = clustering.core_sample_indices_
    # print(centroids)
    # c = []
    # for i in range(0, 9):
    #     c.append(centroids.T[i].sort_values(ascending=False))

    # print('cluster', 'silhouette score', 'dunn index')
    # for i in range (2, 18):
    # i = 6
    # clustering = AgglomerativeClustering(n_clusters=i).fit(X)
    # # Evaluate cluster
    # silhouette_avg = silhouette_score(X, clustering.labels_)
    # dunn_avg = dunn_fast(X, clustering.labels_)
    # print(i, silhouette_avg, dunn_avg)
    # print(centroids)

    # save loaded data
    # folder_path = './data/saved'
    # Path(folder_path).mkdir(parents=True, exist_ok=True)
    # file_name = time.time()
    # df.loc[(df['system_cluster']==0) | (df['system_cluster'] == 3)].to_pickle('{}/{}.pkl.xz'.format(folder_path, file_name)) 
    # bar.next()
    # bar.finish()
    # # for i in np.unique(y[0]):
    # #     print('Cluster', len(y[0][y[0]==i]))

    # print('finish')



    # for i in range(2, 20):
    #     k_means_optimum = KMeans(n_clusters=i, init='k-means++', random_state=0, n_init='auto')
    #     y = k_means_optimum.fit_predict(X)
    #     df['system_cluster'] = y
    #
    #     silhouette_avg = silhouette_score(X, y)
    #     # dunn_avg = dunn_index(X, y)
    #     print(i, silhouette_avg)

    # criterias = k_means_optimum.cluster_centers_
    # percentage_criterias = []
    # for criteria in criterias:
    #     total = sum(criteria)
    #     # total is 100 percent of criteria
    #     # so every value will be divided by total and times 100 for convert decimal value of percentage to percentation
    #     percentage_criteria = [a / total * 100 for a in criteria]
    #     percentage_criterias.append(percentage_criteria)
    #
    # df_percentage_criteria = pd.DataFrame(percentage_criterias, columns=df.columns)

    # print(dunn_avg)
    # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    #
    # model = model.fit(X)
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(model, truncate_mode="level", p=3)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
