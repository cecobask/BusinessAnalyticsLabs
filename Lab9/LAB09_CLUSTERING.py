import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Load dataset.
data = pd.read_csv("lab11_addhealth.csv").dropna()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Uppercase all DataFrame column names.
data.columns = map(str.upper, data.columns)

# Subset clustering variables.
cluster = data[['ALCEVR1', 'MAREVER1', 'ALCPROBS1', 'DEVIANT1', 'VIOL1',
                'DEP1', 'ESTEEM1', 'SCHCONN1', 'PARACTV', 'PARPRES', 'FAMCONCT']]
print(cluster.describe())

# Standardize clustering variables to have mean=0 and sd=1.
clustervar = cluster.copy()
for var in cluster:
    clustervar[var] = preprocessing.scale(clustervar[var].astype('float64'))

# Split data into train and test sets.
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# K-means cluster analysis for 1-9 clusters.
clusters = range(1, 10)
meandist = []
for k in clusters:
    model = KMeans(n_clusters=k).fit(clus_train)
    clusassign = model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1))
                    / clus_train.shape[0])

# Plot the elbow curve.
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()

# Create an object which will contain the results from the cluster analysis with 3 clusters.
model3 = KMeans(n_clusters=3).fit(clus_train)
clusassign = model3.predict(clus_train)

# Perform canonical discriminate analysis.
# Reduce data by creating smaller number of variables that are linear combinations of the 11 clustering vars.
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=model3.labels_, )
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()
