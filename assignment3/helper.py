import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# ML Helpers
def kmeans_variable_clusters(n_centroids, data_x):
    """
    Loop through number of clusters and collect inertia scores
    Create elbow plot
    """
    scores = []
    n_clusters = [i for i in range(1, n_centroids + 1)]
    print("Number of clusters from 1 to", n_centroids, "are being tested")

    for n_cluster in n_clusters:
        from sklearn.cluster import KMeans

        # Build, fit, predict
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        kmeans.fit(data_x)
        scores.append(kmeans.score(data_x) * -1)

    return scores, n_clusters


def gmm_variable_clusters(n_centroids, data_x, covar_type):
    """
    Loop through number of clusters and collect BIC scores
    Create elbow plot
    """
    scores = []
    n_clusters = [i for i in range(1, n_centroids + 1)]
    print("Number of clusters from 1 to", n_centroids, "are being tested")

    for n_cluster in n_clusters:
        from sklearn.cluster import KMeans

        # Build, fit, predict
        gmm = GaussianMixture(n_components=n_cluster,
                              covariance_type=covar_type,
                              random_state=42)
        gmm.fit(data_x)
        scores.append(gmm.bic(data_x))

    return scores, n_clusters

from sklearn.metrics import silhouette_score, silhouette_samples

def kmeans_and_silhouette(n_centroids, data_x):

    """
    Loop through number of clusters and collect silhouette scores
    Create elbow plot
    """

    scores = []
    n_clusters = [i for i in range(2, n_centroids + 1)]
    print("Number of clusters from 1 to", n_centroids, "are being tested")

    for n_cluster in n_clusters:
        from sklearn.cluster import KMeans

        # Build, fit, predict
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        labels = kmeans.fit_predict(data_x)

        silhouette_avg = silhouette_score(data_x, labels)
        scores.append(silhouette_avg)
        print("For n_clusters =", n_cluster,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data_x, labels)

    return scores, n_clusters

def cluster_accuracy(data_y, cluster_labels):
    """
    Adapted from: https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/helpers.py
    """

    pred = np.zeros(data_y.shape)

    for label in set(cluster_labels):
        # Get indicies where labels match
        label_indices = cluster_labels == label

        # Get class labels array corresponding to those indices
        class_array = data_y[label_indices]

        # Count up the number of each distinct label and choose the most common
        # Essentially choosing final label by what is the majority
        # .most_common returns list of tuples (label, count)
        class_label = Counter(class_array).most_common(1)[0][0]

        pred[label_indices] = class_label

    return accuracy_score(data_y, pred)
