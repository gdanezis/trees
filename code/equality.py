import random
from collections import Counter, defaultdict
import numpy as np
import copy
import traceback

from malware import Forest

def split_data(data, length=50):
    'Take a large text and divide it into chunks'
    strings = [data[i:i+length] for i in range(0, len(data) - length, length)]

    string_data = dict([(s, prefixes(s)) for s in strings])
    all_features = defaultdict(int)
    for k,v in string_data.iteritems():
        for vi in v.keys():
            all_features[vi] += 1

    Sz = len(string_data)
    new_features = dict([(f,v) for f,v in all_features.iteritems() if 9 < v < Sz - 9  ])
    #for f, v in new_features.iteritems():
    #    print v, f

    for s in string_data:
        for vi in string_data[s].keys():
            if vi not in new_features:
                del string_data[s][vi]

    spairs = []
    for i in range(int(len(strings) / 2)):
        spairs += [(strings[2*i], strings[2*i+1])]

    random.shuffle(spairs)
    train0 = spairs[:len(spairs) / 2]
    train1 = [(t0, random.choice(train0)[1]) for t0, _ in train0]
    train_labels = [0] * len(train0) + [1] * len(train1)
    train = (train0 + train1, train_labels)

    test0 = spairs[len(spairs) / 2:]
    test1 = [(t0, random.choice(test0)[1]) for t0, _ in test0]
    test_labels = [0] * len(test0) + [1] * len(test1)
    test = (test0 + test1, test_labels)

    return string_data, train, test


def prefixes(string, min_len=3, max_len=10):
    pfx = defaultdict(int)
    for i in range(len(string)):
        for j in range(min_len, max_len):
            if i+j <= len(string):
                pfx[string[i:i+j]] += 1
    return pfx


def process_data(data, pairs, labels):
    items = []
    labs = []
    for (p1, p2), l in zip(pairs, labels):
        assert p1 in data
        assert p2 in data
        items += [p1,      p2]
        labs += [(p2, l), (p1, l)]
    return items, labs


class EqRecords():

    def __init__(self, data, items, labels=None, sID=1):
        self.data = data
        self.items = items
        self.item_set = set(items)
        if labels:
            assert len(items) == len(labels)
        self.labels = labels
        self.sID = sID

    def _filter(self, f, b):
        new_items = [(idx, i) for idx, i in enumerate(self.items) if (f in self.data[i]) == b ]
        new_labels = None
        if self.labels:
            new_labels = [self.labels[idx] for idx, _ in new_items]

        new_items = [i for _, i in new_items]
        return EqRecords(self.data, new_items, new_labels, 2*self.sID + [0, 1][b])


    def size(self):
        return len(self.items)

    def indexes(self):
        return self.items

    def label_distribution(self):
        assert self.labels is not None
        d = {0:0, 1:0}
        for (s, l) in self.labels:
            res = 0 if s in self.item_set else 1
            d[int(res == l)] += 1
        return d, self.sID

    def H(self):
        d, _ = self.label_distribution()
        S = d[1] + d[0]
        if S == 0:
            return -1
        return (float((d[1] - d[0])) / S) - 1.0

    def get_random_feature(self):
        i = random.choice(self.items)
        return random.choice(self.data[i].keys())

    def split_on_feature(self, feature):
        L = self._filter(feature, False)
        R = self._filter(feature, True)

        dH = self.H()
        S = float(self.size())
        dNew = (L.size() / S) * L.H() + (R.size() / S) * R.H()
        return dNew - dH, L, R

def test_init():
    dataEN = file("../data/pg110.txt").read()

    features, train, test = split_data(dataEN, length=1000)
    train_data, train_labels = train

    items, labs = process_data(features, train_data, train_labels)
    rec = EqRecords(features, items, labs)
    assert rec.labels

    assert rec.size() == len(train_data) * 2

    d, _ = rec.label_distribution()

    print d
    print rec.H()

    for _ in range(100):
        f = rec.get_random_feature()
        dh, L, R = rec.split_on_feature(f)
        print "%f\t\"%s\"\t%s" % (dh, f, (L.size(), R.size()))

    F = Forest(trees = 14, numfeatures = 100)
    # R = Record(training_labels, training_records)
    F.train(rec, multicore = False)

    print F.root

if __name__ == "__main__":
    # dataEN = file("../data/pg23428.txt").read()
    # dataFR = file("../data/pg5711.txt").read()