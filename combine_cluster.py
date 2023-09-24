import sys

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np


# Create a function to perform agglomerative clustering and combine minority clusters
def combine_minority_clusters(X, min_cluster_size):
    # Create an instance of AgglomerativeClustering with n_clusters=None for hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')

    # Fit the data to the clustering algorithm
    clustering.fit(X)

    # Access the labels assigned to each data point
    labels = clustering.labels_

    # Count the number of data points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Identify minority clusters
    minority_clusters = unique_labels[counts < min_cluster_size]

    # Create a mapping from original cluster labels to new combined cluster labels
    cluster_mapping = {label: 'combined' for label in minority_clusters}

    # Update the labels to combine minority clusters
    combined_labels = [cluster_mapping[label] if label in minority_clusters else label for label in labels]

    return combined_labels


if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[1])
    X = df.to_numpy()
    # Set a minimum cluster size threshold
    min_cluster_size_threshold = 500

    # Perform clustering and combine minority clusters
    combined_labels = combine_minority_clusters(X, min_cluster_size_threshold)

    # Print the combined cluster labels
    print("Combined Cluster Labels:", combined_labels)
