# About

# This module contains some helpful and commonly used functions

import numpy
import pylab
from sklearn.neighbors.unsupervised import NearestNeighbors

__author__ = 'Alex Rogozhnikov'


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