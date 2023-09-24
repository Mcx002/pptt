import math
import os
import re

import numpy as np
import pandas as pd
import yaml
from jqmcvi.base import dunn_fast
from pandas import DataFrame
from progress.bar import Bar
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt

from config import CORPUS_PATH
from model.document import Document

from kneed import KneeLocator

public_figure = ['anies', 'ganjar', 'prabowo', 'puan', 'ahok', 'kamil', 'ridw', 'maharani', 'maharan', 'subianto',
                 'baswedan', 'pranowo']


def get_file_name(filename):
    split_file = filename.split('.')
    return ''.join(split_file[0:len(split_file) - 1])


def get_ext(filename):
    split_file = filename.split('.')
    return split_file[len(split_file) - 1]


def load_data():
    is_corpus_path_exists = os.path.exists(CORPUS_PATH or '')
    if not is_corpus_path_exists:
        print('path is not found.')

    list_files = os.listdir(CORPUS_PATH)
    list_files = [a for a in list_files if get_ext(a) == 'yaml' or get_ext(a) == 'txt']

    docs: {[str]: Document} = {}

    bar = Bar('Preprocessing Document', max=len(list_files))
    for file in list_files:
        file_path = '{}/{}'.format(CORPUS_PATH, file)
        file_name = get_file_name(file)
        file_ext = get_ext(file)
        is_file_exists = os.path.exists(file_path)
        if not is_file_exists:
            print('file {} not found'.format(file_path))

        doc = Document(file_name)

        if file_name in docs:
            doc = docs[file_name]

        if file_ext == 'txt':
            os_body = open(file_path, 'r')
            body = os_body.read()

            doc.set_body(body)

        if file_ext == 'yaml':
            os_identity = open(file_path, 'r')
            identity = yaml.safe_load(re.sub('[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]', '', os_identity.read()))

            if identity is None:
                continue

            title = '?'
            author = '?'

            if 'title' in identity:
                title = identity['title']

            if 'author' in identity:
                author = identity['author']

            doc.set_title(title)
            doc.set_author(author)

        docs[file_name] = doc
        bar.next()

    bar.finish()
    return [docs[a] for a in docs.keys()]


def text_preprocessing():
    data = load_data()
    doc_n = len(data)
    tfs = {}

    tfs_process_bar = Bar('TFS Process', max=len(data))
    for doc in data:
        for doc_word in doc.tf:
            if doc_word not in tfs:
                tfs[doc_word] = {}
            tfs[doc_word][doc.id] = doc.tf[doc_word]
        tfs_process_bar.next()
    tfs_process_bar.finish()

    dfs = {}
    for tf in tfs:
        dfs[tf] = len(tfs[tf])

    idfs = {}
    for df in dfs:
        idfs[df] = math.log10(doc_n / dfs[df])

    tf_idfs = {}
    for tf in tfs:
        weight = 1
        if tf not in tf_idfs:
            tf_idfs[tf] = {}
        if tf in public_figure:
            weight = 10
        for doc in tfs[tf]:
            tf_idfs[tf][doc] = tfs[tf][doc] * idfs[tf] * weight

    return data, tf_idfs


def generate_tf_idf():
    data, tf_idfs = text_preprocessing()
    result = []
    for tf_idf in tf_idfs:
        item = []
        for doc in data:
            val = 0
            if doc.id in tf_idfs[tf_idf]:
                val = tf_idfs[tf_idf][doc.id]
            item.append(val)
        result.append(item)

    return data, tf_idfs, result


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def inter_cluster_distance(x, y):
    values = np.ones([len(x), len(y)])
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            values[i, j] = euclidean_distance(x[i], y[j])
    return np.min(values)


def intra_cluster_distance(x):
    values = np.zeros([len(x), len(x)])
    for i in range(0, len(x)):
        for j in range(0, len(x)):
            values[i, j] = euclidean_distance(x[i], x[j])
    return np.max(values)


def dunn_index(X, labels):
    clusters = {}
    len_labels = len(labels)
    for i in range(0, len_labels):
        cluster_number = labels[i]
        if cluster_number not in clusters:
            clusters[cluster_number] = []
        clusters[cluster_number].append(X[i])

    if -1 in clusters:
        del clusters[-1]

    inter = np.zeros([len(clusters), len(clusters)])
    intra = np.zeros([len(clusters), 1])
    l_range = list(range(len(clusters)))
    bar = Bar('Dunn index progress', max=len(l_range))
    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            inter[k, l] = inter_cluster_distance(clusters[k], clusters[l])
            intra[k] = intra_cluster_distance(clusters[k])
        bar.next()
    bar.finish()
    without_zero = inter[np.nonzero(inter)]
    return np.min(without_zero) / np.max(intra)


def elbow_method(X, *, min_iter=2, max_iter=10, show_graphic=False, show_knee=False, save_image=""):
    y = []
    x = range(min_iter, max_iter)
    bar = Bar('Preparing Elbow Graph', max=(max_iter-min_iter))
    for i in x:
        k_means = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=0)
        k_means.fit(X)
        y.append(k_means.inertia_)
        bar.next()
    bar.finish()
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.plot(x, y, 'bx-')
    if show_knee:
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')

    if save_image != "":
        plt.savefig(save_image)

    if show_graphic:
        plt.show()

    plt.clf()
    return kn.knee


def silhouette_checking(X, *, min_iter=2, max_iter=10, show_log=False):
    result = 0
    temp_silhouette = -1
    bar = Bar('Silhouette Checking', max=(max_iter-min_iter))
    logs = []
    for i in range(min_iter, max_iter):
        k_means = KMeans(n_clusters=i, init='k-means++', n_init='auto', random_state=0)
        y = k_means.fit_predict(X)

        bar.next()
        silhouette_avg = silhouette_score(X, y)
        logs.append({"cluster": i, "silhouette_score": silhouette_avg})
        if silhouette_avg > temp_silhouette:
            temp_silhouette = silhouette_avg
            result = i
    bar.finish()
    return result, logs


def kmeans(df, X, n_cluster=4):
    k_means_optimum = KMeans(n_clusters=n_cluster, init='k-means++', n_init='auto', random_state=0)
    y = k_means_optimum.fit_predict(X)

    centroids = pd.DataFrame(k_means_optimum.cluster_centers_, columns=df.columns)
    c = []
    for i in range(0, n_cluster):
        c.append(centroids.T[i].sort_values(ascending=False))
    # print(c[0])

    criterias = k_means_optimum.cluster_centers_
    percentage_criterias = []
    for criteria in criterias:
        total = sum(criteria)
        # total is 100 percent of criteria
        # so every value will be divided by total and times 100 for convert decimal value of percentage to percentation
        percentage_criteria = [a / total for a in criteria]
        percentage_criterias.append(percentage_criteria)

    df_percentage_criteria = pd.DataFrame(percentage_criterias, columns=df.columns)

    silhouette_avg = silhouette_score(X, y)
    dunn_avg = dunn_fast(X, y)
    print(n_cluster, silhouette_avg, dunn_avg)
    return y, n_cluster, silhouette_avg, dunn_avg, df_percentage_criteria


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def get_centroid_of_df_cluster(df: DataFrame):
    cnt = len(df.index)
    temp = np.zeros(df.shape[1])
    for i in df.index:
        row = df.loc[i].to_numpy()
        for j in range(len(row)):
            temp[j] += row[j]

    return [x/cnt for x in temp]


def beep():
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))