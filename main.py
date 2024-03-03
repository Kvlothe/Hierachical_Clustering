import pandas as pd
import numpy as np
from data_cleaning import clean_data
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("churn_clean.csv")

x_reference, x_analysis, one_hot_columns, binary_columns, categorical_columns, \
    continuous_columns, continuous_list, df_analysis = clean_data(df)

print('Scaling Data')
print('')
scaler = StandardScaler()
scaled_data = scaler.fit_transform(x_analysis)
print('data scaled')
print('')

# Choose the number of components, e.g., retain 95% of the variance
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(scaled_data)

print(f"Original number of features: {scaled_data.shape[1]}")
print(f"Reduced number of features: {pca_data.shape[1]}")

pca = PCA().fit(scaled_data)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.show()

# # Generate the linkage matrix
# Z = sch.linkage(pca_data, method='ward')
#
# # Generate and plot the dendrogram
# plt.figure(figsize=(10, 7))
# dendrogram = sch.dendrogram(Z)
# plt.title('Dendrogram')
# plt.xlabel('Data Points')
# plt.ylabel('Euclidean distances')
# plt.show()
#
# # Cut the dendrogram at a determined number of clusters, e.g., 7
# clusters = fcluster(Z, 7, criterion='maxclust')
#
# # Add cluster labels to your original dataset for further analysis
# x_analysis['cluster'] = clusters

