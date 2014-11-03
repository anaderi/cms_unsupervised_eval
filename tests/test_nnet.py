"""
Testing all the library
"""
from __future__ import division, print_function
from cmsuml import nnet

__author__ = 'Alex Rogozhnikov'

import numpy


def test_nnet(n_samples=200, n_features=5, distance=0.5):
    from sklearn.datasets import make_blobs
    from sklearn.metrics import roc_auc_score
    X, y = make_blobs(n_samples=n_samples, n_features=5, centers=[numpy.ones(n_features) * distance,
                                                                  - numpy.ones(n_features) * distance])
    for trainer in nnet.trainers:
        for loss in [nnet.squared_loss, nnet.log_loss, nnet.ada_loss]:
            for NNType in [nnet.RBFNeuralNetwork, nnet.SimpleNeuralNetwork,
                           nnet.MultiLayerNetwork, nnet.SoftmaxNeuralNetwork]:
                nn = NNType(layers=[n_features, 5, 1], loss=loss, trainer=trainer)
                nn.fit(X, y)

                print(nn, roc_auc_score(y, nn.predict_proba(X)[:, 1]))

