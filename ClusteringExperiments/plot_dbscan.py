# -*- coding: utf-8 -*-
"""
===================================
DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np

from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import cl_encoder as enc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# #############################################################################
class DBSCANNER(object):
    def __init__(self, X = []):
        self.X = X

    def computeDBSCAN(self, X = None, eps = 0.3, min_samples = 10):
        if X == None:
            X = self.X
        self.eps = eps
        self.min_samples = min_samples
        db = DBSCAN(eps, min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        self.labels = db.labels_
        self.core_samples_mask = core_samples_mask

    def dbscan_metrics(self, labels_true, X = None, labels = None):
        '''
            Create a list of information about the results of a DBSCAN
        '''
        if X == None or not X.any():
            X = self.X
        if labels == None:
            labels = self.labels
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_clusters = n_clusters_
        n_noise_ = list(labels).count(-1)
        info = []

        info.append('Estimated number of clusters: %d' % n_clusters_)
        info.append('Estimated number of noise points: %d' % n_noise_)
        info.append("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        info.append("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        info.append("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        info.append("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        info.append("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        info.append("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))
        self.info = info
        return info

    # #############################################################################
    # Plot result

    def pca_plot_me(self, pca_components = 2, mode = "display", savedir = "figs"):
        X = self.X
        labels = self.labels
        core_samples_mask = self.core_samples_mask

        pca = PCA(pca_components)
        pca.fit(X)
        X = pca.transform(X)

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]

            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)
        plt.title('Estimated number of clusters: %i\neps: %s, min_samples: %i' % (self.n_clusters, "{:.1f}".format(self.eps), self.min_samples))
        if mode == "show":
            plt.show()
        if mode == "save":
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            filename = "DBscan_" + str(self.eps) + "_" + str(self.min_samples) + ".png"
            if savedir:
                filename = savedir + "/" + filename
            plt.savefig(filename)

def main():
    # #############################################################################
    # Read in data

    X, labels_true = enc.vecs_from_CSV("kdd_trimmed.data")

    X, labels_true = enc.encode(X, labels_true)
    X = StandardScaler().fit_transform(X)

    # Compute the DBSCAN results for different hyperparameters
    # A list of lists of labels for the 
    dbscanners = []

    for i in range(1, 5):
        eps = 0.1 * i
        for j in range(5, 26):
            min_samples = j
            db_s = DBSCANNER(X)
            db_s.computeDBSCAN(eps = eps, min_samples = min_samples)
            dbscanners.append(db_s)

    with open("DBScan_results.txt", 'w') as f:
        f.write("DBScan Results\n")
        for db_s in dbscanners:
            info = db_s.dbscan_metrics(labels_true)
            for line in info:
                print(line)
                f.write("-------------------------------\n")
                f.write("eps: " + str(db_s.eps) + "\nmin_samples: " + str(min_samples) + "\n" + line)
            db_s.pca_plot_me(mode = "save")
        f.close()
    

main()