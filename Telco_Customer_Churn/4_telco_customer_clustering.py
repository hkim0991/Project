# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 00:21:02 2018

@author: kimi
"""

# Import libraries & funtions ------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(os.getcwd())
#os.chdir('C:/Users/202-22/Documents/Python - Hyesu/Project/telco')
os.chdir('D:/Data/Python/project')


# Load dataset ----------------------------------------------------------------

train_path = "../data/telco/telco_data_after_onehotencoding.csv"

train = pd.read_csv(train_path, engine='python')

train.shape # 7043 x 39
train.head()
train.isnull().sum() # no missing value


# Dataset preparation ---------------------------------------------------------

final_data = train.copy()
X = final_data
# we don't have y value(target feature) here


# Standarization of the features ----------------------------------------------

all_columns = final_data.columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

df_X_scaled = pd.DataFrame(X_scaled, columns= all_columns)
df_X_scaled.columns
df_X_scaled.head()


# Model 1. K-Means Clustering -------------------------------------------------

from sklearn.cluster import KMeans

# n_clusters = 4
kmeans = KMeans(n_clusters=4, random_state=0).fit(X_scaled)
labels = kmeans.labels_
clust_labels = kmeans.predict(X_scaled) 
cent = kmeans.cluster_centers_
# as we didn't put the new data, labels and clust_labels will have the same value

print("cluster label: \n{}".format(labels))
print("predicted lables: \n{}".format(clust_labels))
print("cluster centers: \n{}".format(cent))


# Add this clustering group in original dataset with categorical features 
# Load original dataset 

data_path = "../data/telco/telco_data_preprocessed.csv"
original = pd.read_csv(data_path, engine='python')

original.shape # 7043 x 21
original.head()

# copy it and drop 'Churn' feature as we don't need it here

clust_data = original.copy()
clust_data.drop('Churn', 1, inplace=True) 

# Insert the cluster group

clust_data['cluster_group'] = clust_labels
clust_data.head()


## writing the final data into a csv file for data processing with SQL --------

import csv
clust_data.describe()
clust_data.info()

clust_data.to_csv('telco_clust_kmeans_n4.csv', index=False)

# Analize the clusters
clust_data.groupby(['cluster_group']).mean()
clust_data['cluster_group'].value_counts()


# Or let's do it at once ! ---------------------------------------------------- 

#n_clusters= range(3, 6)
#for n in n_clusters:
#    kmeans = KMeans(n_clusters=n, random_state=0).fit(X_scaled)
#    clust_label = kmeans.predict(X_scaled)
#    clust_data = train.copy()
#    clust_data['cluster_group'] = clust_label
#    clust_data.to_csv('telco_clust_kmeans_n{}.csv'.format(n), index=False )
 

# Model 2. Agglomerative Clustering -------------------------------------------

from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)


# Dendrogram ------------------------------------------------------------------

from scipy.cluster.hierarchy import dendrogram, ward

linkage_array = ward(X_scaled)

fig = plt.figure(figsize=(12,8))
dendrogram(linkage_array)

# from this dendrogram, we assume that the good number of clustering for this dataset is 4
# Adaptation of the results of dendrogram

agg = AgglomerativeClustering(n_clusters=4).fit_predict(X_scaled)

agg_clust = original.copy()
agg_clust.drop('Churn', 1, inplace=True) 
agg_clust['cluster_group'] = agg

# Analize the clusters
agg_clust.groupby(['cluster_group']).mean()
agg_clust['cluster_group'].value_counts()

# Write a csv file for the process in Hive
agg_clust.to_csv('telco_clust_agg_n4.csv', index=False)


# Model 3. DBSCAN -------------------------------------------------------------

from sklearn.cluster import DBSCAN

# let's try the dbscan algorithm with the default values 
dbscan = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)

import collections, numpy

collections.Counter(dbscan)

# too many noise points (noise points: 5746 out of 7043)... 
# Q. then how can we find the appropriate eps and min_samples values?

# K-distance ------------------------------------------------------------------
import math

def k_distances(X, n=None, dist_func=None):
    """Function to return array of k_distances.

    X - DataFrame matrix with observations
    n - number of neighbors that are included in returned distances (default number of attributes + 1)
    dist_func - function to count distance between observations in X (default euclidean function)
    """
    if type(X) is pd.DataFrame:
        X = X.values
    k=0
    if n == None:
        k=X.shape[1]+2
    else:
        k=n+1

    if dist_func == None:
        # euclidean distance square root of sum of squares of differences between attributes
        dist_func = lambda x, y: math.sqrt(
            np.sum(
                np.power(x-y, np.repeat(2,x.size))
            )
        )

    Distances = pd.DataFrame({
        "i": [i//10 for i in range(0, len(X)*len(X))],
        "j": [i%10 for i in range(0, len(X)*len(X))],
        "d": [dist_func(x,y) for x in X for y in X]
    })
    return np.sort([g[1].iloc[k].d for g in iter(Distances.groupby(by="i"))])

d = k_distances(X_scaled, 3)
plt.plot(d)
plt.ylabel("k-distances")
plt.grid(True)
plt.show()

# # ref.k_distance function: 
# https://stackoverflow.com/questions/43160240/how-to-plot-a-k-distance-graph-in-python


# the final values of eps and min_samples -------------------------------------
# 1. to find out the best eps value:

for e in np.arange(4.5, 5.6, 0.1):
    for m in range(3, 5):
        dbscan = DBSCAN(eps=e, min_samples=m).fit_predict(X_scaled)
        print('eps:', e, '| min_samples', m)
        print('the number of cluster: {}'.format(collections.Counter(dbscan), '\n'))

# 2. to find out the best min_sample value: 
for m in range(10, 100):
        dbscan = DBSCAN(eps=4.8, min_samples=m).fit_predict(X_scaled)
        print('eps: 4.8', '| min_samples', m)
        print('the number of cluster: {}'.format(collections.Counter(dbscan), '\n'))


# Compare the clusters --------------------------------------------------------
# 1. when eps:4.8, MinPt:16
# the number of cluster: Counter({1: 6063, -1: 478, 0: 407, 2: 95})
# % of noise: 478/7043: 6.7%
dbscan1 = DBSCAN(eps=4.8, min_samples=16).fit_predict(X_scaled)            

dbscan_clust1 = original.copy()
dbscan_clust1.drop('Churn', 1, inplace=True) 

dbscan_clust1['cluster_group'] = dbscan1
dbscan_clust1.groupby(['cluster_group']).mean()


# 2. when eps:4.8, MinPt:3
# the number of cluster: Counter({1: 6310, 0: 660, -1: 69, 2: 4})
# noise: 1% but MinPt=3 doesn't reflect the reality... 
dbscan2 = DBSCAN(eps=4.8, min_samples=3).fit_predict(X_scaled)

dbscan_clust2 = original.copy()
dbscan_clust2.drop('Churn', 1, inplace=True) 

dbscan_clust2['cluster_group'] = dbscan2
dbscan_clust2.groupby(['cluster_group']).mean()


# 3. when eps:4.8, MinPt: 83
#the number of cluster: Counter({0: 4360, -1: 2119, 1: 480, 2: 84})
# noise: 30% 
dbscan3 = DBSCAN(eps=4.8, min_samples=83).fit_predict(X_scaled)

dbscan_clust3 = original.copy()
dbscan_clust3.drop('Churn', 1, inplace=True) 

dbscan_clust3['cluster_group'] = dbscan3
dbscan_clust3.groupby(['cluster_group']).mean()


# 4. when eps:4.8, MinPt: 81
#the number of cluster: Counter({0: 4381, -1: 2061, 1: 488, 2: 113})
# noise: 29% 
dbscan4 = DBSCAN(eps=4.8, min_samples=81).fit_predict(X_scaled)

dbscan_clust4 = original.copy()
dbscan_clust4.drop('Churn', 1, inplace=True) 

dbscan_clust4['cluster_group'] = dbscan4
dbscan_clust4.groupby(['cluster_group']).mean()


## comparing means between four different results, 
## #4 seems the one that we want (more distinct 4 groups and less noise points vs dbscan model 3)

# Write a csv file for the process in Hive
dbscan_clust4.to_csv('telco_clust_dbscan_n4.csv', index=False)


# Comparison between three clustering models ----------------------------------
# Spread of mean values of 'TotalCharges' from each clustering model

# k-means(n=4)
kmeans_n4_mean = clust_data.groupby(['cluster_group']).mean().sort_values('TotalCharges')

# agg(n=4)
agg_n4_mean = agg_clust.groupby(['cluster_group']).mean().sort_values('TotalCharges')

# dbscan(n=4)
dbscan_n4_mean = dbscan_clust4.groupby(['cluster_group']).mean().sort_values('TotalCharges')


# Visualization the mean values of 'TotalCharges' from each clustering model
plt.figure(figsize=(15, 7)).subplots_adjust(wspace=0.3)
plt.subplot(1,3,1)
kmeans_n4_mean['TotalCharges'].plot.bar()
plt.title('kmeans')

plt.subplot(1,3,2)
agg_n4_mean['TotalCharges'].plot.bar()
plt.title('agglometrative')

plt.subplot(1,3,3)
dbscan_n4_mean['TotalCharges'].plot.bar()
plt.title('DBSCAN')
plt.show()

