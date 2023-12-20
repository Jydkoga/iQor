import dataprocessing as datpr

# first, getting our data from our filepath to the csv file.
data = datpr.data

# PCA
newdf, pca = datpr.pca(data, num_components=5)

# Each index is the explained percentage of variance in data
# From this, we set the num_components to 5 (because there is marginal gain from including more dimensions)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

datpr.visualize_pca(newdf, "PC1", "PC2")


# K-means with PCA
labels, centroids = datpr.k_means(data, 3, True)
# possible axis 1 and axis 2: AREA_CODE,RSK_LVL,LIMIT,BALANCE,PAYMENT,OCL_AMT,BILL_DAY,CYCL_DLNQ,LAST_AMT,LST_DBT_PMT,LIQ_OFFER_AMT,COLLECTABLE_BAL
datpr.visualize_kmeans(labels, centroids, True, "RSK_LVL", "BALANCE")

# K-means without PCA
labels_no, centroids_no = datpr.k_means(data, 3, False)
datpr.visualize_kmeans(labels_no, centroids_no, False, "RSK_LVL", "BALANCE")

# Using a Correlation Matrix to find correlated labels
datpr.visualize_corr_matrix(data)

# Hierarchical clustering
# we set our threshold to 3 based on the dendrogram, cutting the tallest vertical line.
datpr.hierarchical_clustering(data, 3)

# broken
datpr.h_cluster(data.head(1000))
