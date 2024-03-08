import pandas as pd
import numpy as np
from data_cleaning import clean_data
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import inconsistent
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("churn_clean.csv")

x_reference, x_analysis, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list, df_analysis = clean_data(df)

# Standardizing the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x_analysis)

# Fit PCA to scaled data to determine the number of components
pca = PCA().fit(scaled_data)

# Plot the cumulative explained variance to find a suitable number of components
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()

# Decide the number of components based on the plot
# For example, if 1500 components explain 85% of variance, use n_components=1500
n_components = 0.90  # Adjust this based on the plot
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(scaled_data)

# Perform hierarchical clustering on the PCA-reduced data
Z = sch.linkage(pca_data, method='ward')

# Plot the dendrogram
# plt.figure(figsize=(13, 7))
# dendrogram = sch.dendrogram(Z)
# plt.title('Dendrogram')
# plt.xlabel('Data Points')
# plt.ylabel('Euclidean distances')
# plt.show()

threshold_distance = 230

# Obtain the clusters
clusters = fcluster(Z, threshold_distance, criterion='distance')

# Add cluster labels to your original dataset for further analysis
x_analysis['cluster'] = clusters

plt.title('Truncated Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
sch.dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last 20 merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True  # to get a distribution impression in truncated branches
)
plt.show()

depth = 5  # specify the depth for the calculation
incons = inconsistent(Z, depth)
print(incons[-10:])  # Show the last 10 inconsistency values to aid in decision

# Use the average silhouette method to find the optimal number of clusters
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example range, can be expanded
for n_clusters in range_n_clusters:
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(pca_data, clusters)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Calculating the mean for each feature within each cluster
cluster_means = x_analysis.groupby('cluster').mean()
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
pd.set_option('display.max_rows', None)  # Optional: if you also want to display all rows
print(cluster_means)
cluster_means.to_csv("cluster_means.csv")
for name, group in x_analysis.groupby('cluster'):
    print(f"Cluster {name} Means:")
    print(group.mean())
    print('\n')  # Adds a newline for better readability

for feature in continuous_columns:  # Assuming continuous_columns is a list of your continuous feature names
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y=feature, data=x_analysis)
    plt.title(f'Distribution of {feature} across clusters')
    plt.show()
