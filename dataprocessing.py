import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering

# finding the csv file
curr_directory = os.path.dirname(os.path.abspath(__file__))
file_name = "iqor_data.csv"
file_path = os.path.join(curr_directory, file_name)

# Load data from the CSV file using pandas
data = pd.read_csv(file_path)


# normalize data
def normalize(data):
    # copy the data
    df_max_scaled = data.copy()

    # apply normalization techniques
    for column in df_max_scaled.columns:
        df_max_scaled[column] = (
            df_max_scaled[column] / df_max_scaled[column].abs().max()
        )
    return df_max_scaled


# Input: data, num_components (either integer of decimal (if want to retain 50% of data, set num_components = 0.5))
def pca(data1, num_components):
    scaler = StandardScaler()
    data = scaler.fit_transform(data1)
    # Either choose # featuers or number pca dimensions
    pca = PCA(n_components=num_components)

    pca_result = pca.fit_transform(data)
    new_df = pd.DataFrame(
        data=pca_result, columns=[f"PC{i}" for i in range(1, pca_result.shape[1] + 1)]
    )
    return new_df, pca


# visualizing pca given the output of our pca() function
# input: string dim1, dim2 is PC1, PC2, PC3 ... up to PCn where n = max number of components set in pca() function
def visualize_pca(new_df, dim1, dim2):
    num1 = dim1[-1]
    num2 = dim2[-1]
    # Scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=dim1, y=dim2, data=new_df)
    plt.title(f"PCA Results - {dim1} vs {dim2}")
    plt.xlabel(f"Principal Component {num1} ({dim1})")
    plt.ylabel(f"Principal Component {num2} ({dim2})")
    plt.show()


# Input: data, int k = num centroids, boolean usePCA, int num_components for PCA (Default 5),
# (int) max_iterations for number of iterations k-means runs (default 100), float tolerance on when to stop running k-means (default 1e-4)
def k_means(data, k, usePCA, num_components=5, max_iterations=100, tol=1e-4):
    # if we want to use k_means with PCA
    if usePCA:
        data, _ = pca(data, num_components)

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


# input: labels and centroids from k_means function. usePCA is a boolean that is set to true if using PCA.
# Define two strings, axis 1 and axis 2 corresponding to which labels you want to show on the plot.
def visualize_kmeans(labels, centroids, usePCA, axis1, axis2):
    # possible dimensions: AREA_CODE,RSK_LVL,LIMIT,BALANCE,PAYMENT,OCL_AMT,BILL_DAY,CYCL_DLNQ,LAST_AMT,LST_DBT_PMT,LIQ_OFFER_AMT,COLLECTABLE_BAL
    # visualize first two dimensions of clusters
    plt.scatter(
        data[axis1],
        data[axis2],
        c=labels,
        cmap="viridis",
        s=50,
        alpha=0.8,
        edgecolors="w",
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids"
    )
    if usePCA:
        plt.title("K-Means Clustering Results with PCA")
    else:
        plt.title("K-Means Clustering Results without PCA")
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.legend()
    plt.show()


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
    # setting threshold to determine number of clusters
    threshold = numClu

    # plotting dendrogram
    dn = hierarchy.dendrogram(
        distance_matrix, labels=data.columns, color_threshold=threshold
    )
    plt.show()

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
    return labels


# without using correlation matrix
# from https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/3b.%20Hierarchical%20Clustering.ipynb
def h_cluster(data):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Perform hierarchical clustering
    hier_cluster = AgglomerativeClustering(
        affinity="euclidean", linkage="ward", compute_full_tree=True
    )
    hier_cluster.set_params(n_clusters=2)
    clusters = hier_cluster.fit_predict(data_scaled)

    # Add cluster labels to the original data
    data_clustered = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
    data_clustered["cluster"] = clusters

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # Visualize the clustered data
    display_factorial_planes(
        data_pca, 2, pca, [(0, 1)], illustrative_var=clusters, alpha=0.8
    )


# pasted from: https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis/blob/master/functions.py
def display_factorial_planes(
    X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None
):
    """Display a scatter plot on a factorial plane, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks:
        if d2 < n_comp:
            # Initialise the matplotlib figure
            fig = plt.figure(figsize=(7, 6))

            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(
                        X_projected[selected, d1],
                        X_projected[selected, d2],
                        alpha=alpha,
                        label=value,
                    )
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i, (x, y) in enumerate(X_projected[:, [d1, d2]]):
                    plt.text(x, y, labels[i], fontsize="14", ha="center", va="center")

            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
            plt.xlim([-boundary, boundary])
            plt.ylim([-boundary, boundary])

            # Display grid lines
            plt.plot([-100, 100], [0, 0], color="grey", ls="--")
            plt.plot([0, 0], [-100, 100], color="grey", ls="--")

            # Label the axes, with the percentage of variance explained
            plt.xlabel(
                "PC{} ({}%)".format(
                    d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)
                )
            )
            plt.ylabel(
                "PC{} ({}%)".format(
                    d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)
                )
            )

            plt.title("Projection of points (on PC{} and PC{})".format(d1 + 1, d2 + 1))
            plt.show()


def visualize_hcluster(data):
    data_norm = normalize(data)
    pca = PCA(n_components=2)
    pca.fit(data_norm)
    visualize_pca(data_norm, "PC1", "PC2")

    data_reduced = pca.transform(data_norm)

    labels = hierarchical_clustering(data, numClu=2)
    print("i got to the point right after hierarchical_clustering")

    display_factorial_planes(
        data_reduced, 2, pca, [(0, 1)], illustrative_var=labels, alpha=0.8
    )
