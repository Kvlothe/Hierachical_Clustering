import pandas as pd
import numpy as np
from data_cleaning import clean_data
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import inconsistent
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math


def boxplots_grouped(df, continuous_columns, folder_path='Box Plot Grouped', plots_per_page=6):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Number of pages
    num_plots = len(continuous_columns)
    num_pages = math.ceil(num_plots / plots_per_page)

    # Create plots
    for page in range(num_pages):
        plt.figure(figsize=(18, 12))  # Adjust the size as needed
        for i in range(min(plots_per_page, num_plots - page * plots_per_page)):
            plt.subplot(math.ceil(plots_per_page / 2), 2, i + 1)
            feature = continuous_columns[page * plots_per_page + i]
            sns.boxplot(x='cluster', y=feature, data=df)
            plt.title(f'Distribution of {feature} across clusters')
        plt.tight_layout()
        plt.savefig(f'{folder_path}/Boxplot_Page_{page + 1}.png')
        plt.close()


df = pd.read_csv("churn_clean.csv")

x_reference, x_analysis, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list, df_analysis = clean_data(df)

# Standardizing the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x_analysis)

# Fit PCA to scaled data to determine the number of components
pca = PCA().fit(scaled_data)

# Plot the cumulative explained variance to find a suitable number of components
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by Components')
# plt.show()

# Decide the number of components based on the plot
n_components = 0.95  # Adjust this based on the plot
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

threshold_distance = 250

# Obtain the clusters
clusters = fcluster(Z, threshold_distance, criterion='distance')

# Add cluster labels to your original dataset for further analysis
x_analysis['cluster'] = clusters

# plt.title('Truncated Dendrogram')
# plt.xlabel('Cluster Size')
# plt.ylabel('Distance')
sch.dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=20,  # show only the last 20 merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True  # to get a distribution impression in truncated branches
)
# plt.show()

depth = 5  # specify the depth for the calculation
incons = inconsistent(Z, depth)
print(incons[-10:])  # Show the last 10 inconsistency values to aid in decision

# Use the average silhouette method to find the optimal number of clusters
range_n_clusters = [2, 3, 4, 5, 6]  # Example range, can be expanded
for n_clusters in range_n_clusters:
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    silhouette_avg = silhouette_score(pca_data, clusters)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Calculate Davies-Bouldin Index
# Lower Davies-Bouldin index relates to a model with better separation between the clusters
davies_bouldin = davies_bouldin_score(pca_data, clusters)
print(f"Davies-Bouldin Index: {davies_bouldin}")

# Calculate Calinski-Harabasz Index
# Higher Calinski-Harabasz score relates to a model with better defined clusters
calinski_harabasz = calinski_harabasz_score(pca_data, clusters)
print(f"Calinski-Harabasz Index: {calinski_harabasz}")

# Calculating the mean for each feature within each cluster
cluster_means = x_analysis.groupby('cluster').mean()
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
pd.set_option('display.max_rows', None)  # Optional: if you also want to display all rows
# print(cluster_means)
cluster_means.to_csv("cluster_means.csv")
# for name, group in x_analysis.groupby('cluster'):
#     print(f"Cluster {name} Means:")
#     print(group.mean())
#     print('\n')  # Adds a newline for better readability

boxplots_grouped(x_analysis, continuous_columns)

