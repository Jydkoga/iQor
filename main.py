import dataprocessing as datpr
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# new_df, pca = principal_component_analysis(data, num_components=5)

# Each index is the explained percentage of variance in data
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)

data = datpr.data
k = 4
labels, centroids = datpr.k_means(data, k)

print("Final Centroids:\n", centroids)
print("Labels:\n", labels)

# possible dimensions: AREA_CODE,RSK_LVL,LIMIT,BALANCE,PAYMENT,OCL_AMT,BILL_DAY,CYCL_DLNQ,LAST_AMT,LST_DBT_PMT,LIQ_OFFER_AMT,COLLECTABLE_BAL
# visualize first two dimensions of clusters
plt.scatter(
    data["OCL_AMT"],
    data["BALANCE"],
    c=labels,
    cmap="viridis",
    s=50,
    alpha=0.8,
    edgecolors="w",
)
plt.scatter(
    centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids"
)
plt.title("K-Means Clustering Results without PCA")
plt.xlabel("OCL_AMT")
plt.ylabel("BALANCE")
plt.legend()
plt.show()

datpr.visualize_corr_matrix(data)
