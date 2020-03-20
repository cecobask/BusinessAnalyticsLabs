import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Load dataset.
data = pd.read_csv("../gapminder.csv").dropna()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

# Uppercase all DataFrame column names.
data.columns = map(str.upper, data.columns)

# Subset clustering variables and copy to a new variable.
cluster = data.loc[:, ['INCOMEPERPERSON', 'FEMALEEMPLOYRATE', 'INTERNETUSERATE',
                       'LIFEEXPECTANCY', 'ALCCONSUMPTION', 'URBANRATE']].copy()

# Loop through every variable in the cluster.
for var in cluster:
    cluster[var] = cluster[var].replace(' ', np.NaN)  # Check for empty values and replace with NaN.
    cluster[var] = pd.to_numeric(cluster[var])  # Convert each column to a number.
    cluster[var] = preprocessing.scale(cluster[var].astype('float64'))  # Standardize clustering variables.

# Split the data into train and test sets.
clus_train, clus_test = train_test_split(cluster, test_size=.3, random_state=123)

clus_train.fillna(clus_train.mean(), inplace=True)  # Fill NaN values with a mean number.

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

"""
We can see that the average distance decreases as the number of clusters increases.
There appears to be a couple of bends at the line at 2 clusters and at 3 clusters, but it's not very clear.
Further investigation is needed.
"""

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

"""
The scatter plot shows that the 3 clusters are clearly separated, with minimal overlap between them.
The only overlap is apparent between the middle cluster with its neighbouring clusters in the top parts.
The observations are well spread out within the confines of their clusters.
This indicates less correlation among the observations and higher within the cluster variance.
The results suggest that the two-cluster solution might be better,
as it might provide better separation between the clusters.
Further investigation is needed, however time constraints do not allow it.
"""