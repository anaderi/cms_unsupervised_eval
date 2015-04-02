"""
Testing all the library
"""
from __future__ import division, print_function
from sklearn.linear_model.logistic import LogisticRegression
from cmsuml import nnet
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

__author__ = 'Alex Rogozhnikov'

import numpy


def test_nnet(n_samples=200, n_features=5, distance=0.5):
    X, y = make_blobs(n_samples=n_samples, n_features=5, centers=[numpy.ones(n_features) * distance,
                                                                  - numpy.ones(n_features) * distance])

    NNTypes = [
        nnet.SimpleNeuralNetwork,
        nnet.MultiLayerNetwork,
        nnet.SoftmaxNeuralNetwork,
        nnet.RBFNeuralNetwork,
        nnet.PairwiseNeuralNetwork,
        nnet.PairwiseSoftplusNeuralNetwork,
        # nnet.ObliviousNeuralNetwork,
    ]

    for loss in [nnet.log_loss,
                 nnet.ada_loss,
                 nnet.squared_loss,
                 ]:
        for NNType in NNTypes:
            for trainer in nnet.trainers:
                nn = NNType(layers=[n_features, 5, 1], loss=loss, trainer=trainer)
                nn.fit(X, y, stages=100, verbose=nnet.SILENT)
                print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)

    lr = LogisticRegression().fit(X, y)
    print(lr, roc_auc_score(y, lr.predict_proba(X)[:, 1]))

    assert 0 == 1

test_nnet()

# def test_oblivious(n_samples=200, n_features=5, distance=0.5):
#     nn = nnet.ObliviousNeuralNetwork(layers=[n_features, 3, 1], trainer='irprop')
#     X, y = make_blobs(n_samples=n_samples, n_features=5, centers=[numpy.ones(n_features) * distance,
#                                                                   - numpy.ones(n_features) * distance])
#     nn.fit(X, y, batch=100, verbose=10)
#     print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)



