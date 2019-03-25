"""
===================================
K-Means clustering algorithm
===================================

"""

import numpy as np

from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import cl_encoder as enc

# #############################################################################
# Read in data

X, labels_true = enc.vecs_from_CSV("kdd_trimmed.data")

X, labels_true = enc.encode(X, labels_true)
X = StandardScaler().fit_transform(X)

# #############################################################################
# Do the thing

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_


# #############################################################################
# Plot result
import matplotlib.pyplot as plt



