import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from scipy.linalg import eigh

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


new_df, pca = principal_component_analysis(data, num_components=5)
# Each index is the explained percentage of variance in data
print("Explained Variance Ratio:", pca.explained_variance_ratio_)


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


k = 4
labels, centroids = k_means(new_df, k)

print("Final Centroids:\n", centroids)
print("Labels:\n", labels)

# visualize first two dimensions of clusters
plt.scatter(
    new_df["PC1"],
    new_df["PC2"],
    c=labels,
    cmap="viridis",
    s=50,
    alpha=0.8,
    edgecolors="w",
)
plt.scatter(
    centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids"
)
plt.title("K-Means Clustering Results")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
