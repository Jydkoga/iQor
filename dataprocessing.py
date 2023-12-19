import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from scipy.linalg import eigh
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance

curr_directory = os.path.dirname(os.path.abspath(__file__))

file_name = "iqor_data.csv"

file_path = os.path.join(curr_directory, file_name)

# Load data from the CSV file using pandas
data = pd.read_csv(file_path)


# Input: data, num_components (either integer of decimal (if want to retain 50% of data, set num_components = 0.5))
def principal_component_analysis(data, num_components):
    # standardizing features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Either choose # featuers or number pca dimensions
    pca = PCA(n_components=num_components)

    pca_result = pca.fit_transform(scaled_data)
    new_df = pd.DataFrame(
        data=pca_result, columns=[f"PC{i}" for i in range(1, pca_result.shape[1] + 1)]
    )
    return new_df, pca


def k_means(data, k, max_iterations=100, tol=1e-4):
    # randomly initialize centroids
    centroids = data.values[np.random.choice(len(data), k, replace=False)]

    for _ in range(max_iterations):
        # assigning each data point to nearest centroid
        labels = np.argmin(
            np.linalg.norm(data.values - centroids[:, np.newaxis], axis=2), axis=0
        )

        # update centroids based off of mean of assigned data points
        new_centroids = np.array(
            [
                data.values[labels == i].mean(axis=0)
                if np.sum(labels == i) > 0
                else centroids[i]
                for i in range(k)
            ]
        )

        # convergence check
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
    return labels, centroids


# Implementing a correlation matrix to see the correlation between data in the csv file
def get_corr_matrix(data):
    return data.corr()


def visualize_corr_matrix(data):
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.show()


# Using the correlation matrix, we use the hierarchical clustering algorithm instead of k-means
# input: data, number of clusters. In our case, we set numClu to 2 (from dendrogram graph)
def hierarchical_clustering(data, numClu):
    # Finding distances between values in the correlation matrix
    correlation_values = np.asarray(get_corr_matrix(data).values)
    distance_matrix = hierarchy.linkage(
        distance.pdist(correlation_values), method="ward"
    )

    # plotting dendrogram
    dn = hierarchy.dendrogram(distance_matrix, labels=data.columns)
    plt.show()

    # setting threshold to determine number of clusters
    threshold = numClu

    labels = hierarchy.fcluster(distance_matrix, threshold, criterion="distance")

    # Print the columns grouped by their clusters
    clustered_columns = {}
    for col, label in zip(data.columns, labels):
        if label not in clustered_columns:
            clustered_columns[label] = [col]
        else:
            clustered_columns[label].append(col)

    for cluster, columns in clustered_columns.items():
        print(f"Cluster {cluster}: {columns}")
