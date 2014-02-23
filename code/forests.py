## This is an educational random forest implementation

## References:
## * A. Criminisi, J. Shotton, and E. Konukoglu, Decision Forests: 
##   A Unified Framework for Classification, Regression, Density Estimation, 
##   Manifold Learning and Semi-Supervised Learning. Foundations and Trends in 
##   Computer Graphics and Computer Vision. NOW Publishers. Vol.7: No 2-3, pp 81-227. 2012.
##
## * Jamie Shotton, Toby Sharp, Pushmeet Kohli, Sebastian Nowozin, John Winn, 
##   and Antonio Criminisi, Decision Jungles: Compact and Rich Models for 
##   Classification, in Proc. NIPS, 2013

import random
from collections import Counter
import numpy as np
import copy

def split_data(data, label=0, length=50):
    'Take a large text and divide it into chunks'
    strings = [data[i:i+length] for i in range(0, len(data) - length, length)]
    random.shuffle(strings)
    strings = [(s, label) for s in strings]

    test = strings[:len(strings) * 10 / 100]
    training = strings[len(strings) * 10 / 100:]
    return test, training


def entropy(data):
    'Computes the binary entropy of labelled data'
    v = Counter([b for _, b in data]).values()
    d = np.array(v) / float(sum(v))
    return - sum(d * np.log(d))


def split(train, feat):
    'Split data according to an infromation gain criterium'
    ## first compute the entropy
    Hx = entropy(train)
    if Hx < 0.000001:
        raise Exception("Entropy very low")
    L1 = []
    L2 = []
    for t in train:
        if feat in t[0]:
            L1 += [t]
        else:
            L2 += [t]

    E1 = entropy(L1)
    E2 = entropy(L2)
    L = float(len(train))

    H = Hx - E1 * len(L1)/L - E2 * len(L2)/L
    return H, L1, L2, feat

## --------------------------
## - The random forest code -
## --------------------------


def build_tree(train, features, levels=5, numfeatures=100):
    'Train a decsision tree based on labeled data and features'
    if levels == 0:
        C1 = Counter([b for _, b in train])
        Leaf = (None, C1)
        return Leaf
    else:
        try:
            X = (split(train, F) for F in random.sample(features, numfeatures))
            H, L1, L2, F = max(X)
            M1 = build_tree(L1, features, levels - 1, numfeatures)
            M2 = build_tree(L2, features, levels - 1, numfeatures)
            Branch = (F, M1, M2)
            return Branch
        except:
            return build_tree(train, features, levels=0)


def classify(tree, item):
    'Get a decision for an item using a tree'
    if len(tree) == 2:
        assert tree[0] is None
        return tree[1]
    else:
        fet, L1, L2 = tree
        if fet in item:
            return classify(L1, item)
        else:
            return classify(L2, item)

## ----------------------------
## - The decision jungle code -
## ----------------------------


def build_jungle(train, features, levels=10, numfeatures=100):
    DAG = {0: copy.copy(train)}
    Candidate_sets = [0]
    next_ID = 0
    M = 10

    for level in range(levels):
        result_sets = []
        for tdata_idx in Candidate_sets:
            tdata = DAG[tdata_idx]

            if entropy(tdata) == 0.0:
                next_ID += 1
                idx1 = next_ID
                result_sets += [idx1]
                DAG[idx1] = tdata + []
                del DAG[tdata_idx][:]
                DAG[tdata_idx] += [True, idx1, idx1]
                continue

            X = (split(tdata, F) for F in random.sample(features, numfeatures))
            H, L1, L2, F = max(X)

            # Branch = (F, M1, M2)
            next_ID += 1
            idx1 = next_ID
            DAG[idx1] = L1
            next_ID += 1
            idx2 = next_ID
            DAG[idx2] = L2

            result_sets += [idx1, idx2]
            del DAG[tdata_idx][:]
            DAG[tdata_idx] += [F, idx1, idx2]

        ## Now optimize the result sets here
        random.shuffle(result_sets)

        basic = result_sets[:M]
        for r in result_sets[M:]:
            maxv = None
            maxi = None
            for b in basic:
                L = float(len(DAG[r] + DAG[b]))
                e1 = len(DAG[r]) * entropy(DAG[r])
                e2 = len(DAG[b]) * entropy(DAG[b])
                newe = L * entropy(DAG[r] + DAG[b])
                score = abs(e1 + e2 - newe)
                if maxv is None:
                    maxv = score
                    maxi = b
                    continue
                if score < maxv:
                    maxv = score
                    maxi = b
            DAG[maxi] += DAG[r]
            del DAG[r]
            DAG[r] = DAG[maxi]

        Candidate_sets = basic

    for tdata_idx in Candidate_sets:
        tdata = DAG[tdata_idx]
        C1 = Counter([b for _, b in tdata])
        del DAG[tdata_idx][:]
        DAG[tdata_idx] += [None, C1]

    return DAG


def classify_jungle(DAG, item):
    branch = DAG[0]
    while branch[0] is not None:
        try:
            fet, L1, L2 = branch
            if fet == True or fet in item:
                branch = DAG[L1]
            else:
                branch = DAG[L2]
        except:
            print len(branch)
            raise
    return branch[1]

## -------------------------
## - Sample classification -
## -------------------------

if __name__ == "__main__":
    dataEN = file("../data/pg23428.txt").read()
    dataFR = file("../data/pg5711.txt").read()

    length = 50

    testEN, trainEN = split_data(dataEN, label=0, length=length)
    testFR, trainFR = split_data(dataFR, label=1, length=length)

    print "training: EN=%s FR=%s" % (len(trainEN), len(trainFR))

    train = trainEN + trainFR
    random.shuffle(train)
    test = testEN + testFR
    random.shuffle(test)

    ## Now make a bunch of features
    ## A feature is in at least 10% of strings
    ## but also at most in 90% of strings

    sometrain = random.sample(train, 1000)
    features = set()
    while len(features) < 700:
        fragment, _ = random.choice(sometrain)
        l = int(round(random.expovariate(0.20)))
        b = random.randint(0, max(0, length - l))
        feat = fragment[b:b+l]

        ## Test
        C = 0
        for st, _ in sometrain:
            if feat in st:
                C += 1

        f = float(C) / 1000
        if f > 0.01 and f < 0.99 and feat not in features:
            features.add(feat)

    features = list(features)

    manytrees = []
    jungle = []
    for i in range(10):
        print "Build tree %s" % i
        size = len(train) / 3
        training_sample = random.sample(train, size)

        tree = build_jungle(training_sample, features, numfeatures=100)
        jungle += [tree]

        tree = build_tree(training_sample, features, numfeatures=100)
        manytrees += [tree]

    testdata = test
    results_tree = Counter()
    results_jungle = Counter()
    for item, cat in testdata:
        # Trees
        c = Counter()
        for tree in manytrees:
            c += classify(tree, item)
        res = (max(c, key=lambda x: c[x]), cat)
        results_tree.update([res])

        # Jungle
        c = Counter()
        for tree in jungle:
            c += classify_jungle(tree, item)
        res = (max(c, key=lambda x: c[x]), cat)
        results_jungle.update([res])

    print
    print "Results         Tree   Jungle"
    print "True positives:  %4d    %4d" \
        % (results_tree[(1, 1)], results_jungle[(1, 1)])
    print "True negatives:  %4d    %4d" \
        % (results_tree[(0, 0)], results_jungle[(0, 0)])
    print "False positives: %4d    %4d" \
        % (results_tree[(1, 0)], results_jungle[(1, 0)])
    print "False negatives: %4d    %4d" \
        % (results_tree[(0, 1)], results_jungle[(0, 1)])
