import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering, KMeans

# Always make a fuss in case of numerical error
np.seterr(all="raise")


## Helper functions
## Estimate params
def estimate_norm(datas):
    if datas.shape[0] < 2:
        return None, None, 0.0

    mp = np.mean(datas, axis=0)
    sp = np.cov(datas.transpose())

    sign, logdet = np.linalg.slogdet(sp)
    if np.isnan(logdet) or np.isinf(logdet):
        return mp, sp, 0.0

    ent = sign * logdet
    return mp, sp, ent


def split(data, feature, old_ent):
    len_data = float(data.shape[0])
    axis, value = feature
    L1 = data[data[:, axis] <= value]
    L2 = data[data[:, axis] > value]
    _, _, entL1 = estimate_norm(L1)
    _, _, entL2 = estimate_norm(L2)

    ent = (L1.shape[0] / len_data) * entL1 + \
          (L2.shape[0] / len_data) * entL2
    D_ent = old_ent - ent
    return D_ent, L1, L2, feature


## Constants
PROC, BRANCH, LEAF = 0, 1, 2

## Train a clustering tree
def make_tree(datas, feature):
    tree = [(PROC, datas)]
    process = [0]

    while len(process) > 0:
        next_item = process.pop(0)
        xtype, dat = tree[next_item]
        if dat.shape[0] < 50:
            tree[next_item] = (LEAF, dat)
            continue

        assert xtype == PROC
        _, _, old_ent = estimate_norm(dat)

        sample_features = random.sample(feature, 100)
        lot = [split(dat, f, old_ent) for f in sample_features]
        ret = max(lot, key=lambda x: x[0])
        D_ent, L1, L2, feat = ret

        if D_ent < 1.0:
            tree[next_item] = (LEAF, dat)
            continue

        newID_L1 = len(tree)
        tree += [(PROC, L1)]
        newID_L2 = newID_L1 + 1
        tree += [(PROC, L2)]
        process += [newID_L1, newID_L2]

        tree[next_item] = (BRANCH, feat, L1, L2)
    return tree


# Set some undelying structure here
points = [(0, 2),
          (1, 1),
          (2, 0),
          (2, 2),
          (3, 1)]

## The spread of the samples around the points
var = 0.03
cov = np.array([[var, 0], [0, var]])

# Generate some synthetic data around the points
datas = None
for p in points:
    samples = np.random.multivariate_normal(p, cov, 100)
    if datas is None:
        datas = samples
    else:
        datas = np.concatenate([datas, samples])

# Add a splat across all data points to simulate noise
mu, sig, _ = estimate_norm(datas)
splat = np.random.multivariate_normal(mu, sig, 500)
datas = np.concatenate([datas, splat])

# Make up some features we could split on
feature = []
len_datas = datas.shape[0]
for _ in range(1000):
    i = random.randint(0, len_datas-1)
    j = random.choice([0, 1])
    feature += [(j, datas[i, j])]


def tokey(x):
    return tuple(x)

## Make a profile for each data point
profiles = {}
keys = []
for i in range(datas.shape[0]):
    k = tokey(datas[i, :])
    keys += [k]
    profiles[k] = []

## Train a number of clustering trees
NUM_TREES = 200
for j in range(NUM_TREES):
    print "Training tree %s" % j
    t = make_tree(datas, feature)
    cluster_id = 0
    for item in t:
        if item[0] == LEAF:
            dat = item[1]
            for i in range(dat.shape[0]):
                k = tokey(dat[i, :])
                profiles[k] += [cluster_id]
            cluster_id += 1

## Build a affinity matrix from co-occupancy of clusters
for p in profiles:
    profiles[p] = np.array(profiles[p], dtype=int)

covar = np.zeros((len(keys), len(keys)))
for ik1, k1 in enumerate(keys):
    for ik2, k2 in enumerate(keys):
        D = float(np.sum(profiles[k1] == profiles[k2])) / NUM_TREES
        covar[ik1, ik2] = D

## Perform clustering on the affinity matrix
clustering = SpectralClustering(affinity="precomputed", n_clusters=5)
X = clustering.fit(covar)

# Example of using K-means
# clustering = KMeans(n_clusters=5)
# X = clustering.fit(covar)

## Make a picture
colors = ["r", "g", "b", "k", "c", "y", "m"]
cols = [colors[i % 7] for i in X.labels_]
plt.scatter(datas[:, 0], datas[:, 1], c=cols)
plt.show()
