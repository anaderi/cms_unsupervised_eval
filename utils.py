# About

# This module contains some helpful and commonly used functions

import numpy
import pandas
import pylab
import sklearn
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin

__author__ = 'Alex Rogozhnikov'


def get_file_labels(labels, folder='/mnt/w76/notebook/datasets/doudko/samples/'):
    file_labels = {}
    for name in os.listdir(folder):
        for i, label in enumerate(labels):
            if label in name:
                file_labels[name] = i
    return file_labels


def load_data(file_labels, variables, folder='/mnt/w76/notebook/datasets/doudko/samples/', weight_column='weight'):
    import root_numpy
    all_variables = variables + [weight_column]
    data_parts = []
    data_labels = []
    data_weights = []
    for filename, label in file_labels.iteritems():
        data = root_numpy.root2array(folder + filename, treename='Vars', branches=all_variables)
        data = pandas.DataFrame(data)
        data_labels.append(numpy.zeros(len(data), dtype=int) + label)
        data_weights.append(data[weight_column])
        data_parts.append(data.drop([weight_column], axis=1))

    data = pandas.concat(data_parts, ignore_index=True)
    labels = numpy.concatenate(data_labels)
    weights = numpy.concatenate(data_weights)
    return data, labels, weights


def check_sample_weight(y_true, sample_weight):
    """Checks the weights, returns normalized version """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The lengths are different: {0} and {1}".format(len(y_true), len(sample_weight))
        return sample_weight


def distance_quality_matrix(X, y, n_neighbors=50):
    """On of the ways to measure the quality of knning"""
    labels = numpy.unique(y)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    knn_indices = nn.kneighbors(X, n_neighbors=n_neighbors, return_distance=False)
    confusion_matrix = numpy.zeros([len(labels), len(labels)], dtype=int)
    for label1, labels2 in zip(y, numpy.take(y, knn_indices)):
        for label2 in labels2:
            confusion_matrix[label1, label2] += 1

    return confusion_matrix


def plot_confusion_matrix(matrix):
    pylab.figure(figsize=(16, 4))
    pylab.subplot(131), pylab.title('Original')
    matrix = numpy.array(matrix, dtype=float)
    pylab.pcolor(matrix / matrix.sum(axis=None)), pylab.colorbar()
    pylab.subplot(132), pylab.title('Normed by row')
    pylab.pcolor(matrix / matrix.sum(axis=1, keepdims=True)), pylab.colorbar()
    pylab.subplot(133), pylab.title('Normed by column')
    pylab.pcolor(matrix / matrix.sum(axis=0, keepdims=True)), pylab.colorbar()


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, base_estimator):
        self.base_estimator = sklearn.clone(base_estimator)

    def fit(self, X, y, sample_weight):
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        self.estimators = dict()
        labels = numpy.unique(y)
        X = numpy.array(X)
        for label1, label2 in itertools.combinations(labels, 2):
            mask = numpy.in1d(y, [label1, label2])
            clf = sklearn.clone(self.base_estimator)
            clf.fit(X[mask, :], y[mask], sample_weight=sample_weight[mask])
            self.estimators[(label1, label2)] = clf
            print('trained', label1, 'vs', label2)

    def transform(self, X, keep_original_features=True):
        if keep_original_features:
            result = pandas.DataFrame.copy(X)
        else:
            result = pandas.DataFrame()
        for (label1, label2), clf in self.estimators.iteritems():
            result['trained%ivs%i' % (label1, label2)] = clf.predict_proba(X)[:, 1]

        return result