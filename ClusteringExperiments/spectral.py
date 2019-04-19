#!/usr/bin/env python3
"""
===================================
Spectral clustering algorithm
===================================

"""
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np

from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import cl_encoder as enc

from ML.OOD import mysql

# #############################################################################
# Read in data

X, labels_true = enc.vecs_from_CSV("kdd_trimmed.data")

X, labels_true = enc.encode(X, labels_true)
X = StandardScaler().fit_transform(X)

num_labels = (len(set(labels_true)))

# #############################################################################
# Do the thing

spectral = SpectralClustering(n_clusters=2,
                assign_labels="discretize",
                random_state=0).fit(X)

y_pred = kmeans.labels_

y_true = [1 if x > 0 else 0 for x in labels_true]

correct = sum(np.multiply(y_pred, y_true))
print("Accuracy for Spectral Clustering with k = 2 on attacks vs. normal: ", correct/len(y_pred))


spectral = SpectralClustering(n_clusters=num_labels,
                assign_labels="discretize",
                random_state=0).fit(X)
y_pred = kmeans.labels_

y_true = labels_true

correct = sum(np.multiply(y_pred, y_true))
print("Accuracy for Spectral Clustering with k =", num_labels, "for any attack type with >=50 samples and normal data: ", correct/len(y_pred))
