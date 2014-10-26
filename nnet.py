from __future__ import print_function, division
import numpy
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator, ClassifierMixin
import utils
floatX = theano.config.floatX
__author__ = 'Alex Rogozhnikov'

# This is simple version of neuralNetworks, which can be easily adopted.


def squared_loss(y, pred, w):
    return T.mean(w * (y - pred) ** 2)


def log_loss(y, pred, w):
    return -T.mean(w * (y * T.log(pred) + (1 - y) * T.log(1 - pred)))


# TODO think of dropper and noises
class AbstractNeuralNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, learning_rate=0.01, loss=squared_loss, stages=1000, batch=90):
        self.layers = layers
        self.learning_rate = learning_rate
        self.stages = stages
        self.batch = batch
        self.loss = loss
        self.prepared = False
        self.parameters = {}
        self.derivatives = {}
        # Dirty hack for AdaBoost
        self.classes_ = numpy.array([0, 1])

    def prepare(self):
        # This method should be overloaded by descendant
        raise NotImplementedError()

    def _prepare(self):
        activation = self.prepare()
        loss_ = lambda x, y, w: self.loss(y, activation(x), w)
        x = T.matrix('X')
        y = T.vector('y')
        w = T.vector('w')
        self.Activation = theano.function([x], activation(x))
        self.Loss = theano.function([x, y, w], loss_(x, y, w))
        for name, param in self.parameters.iteritems():
            self.derivatives[name] = theano.function([x, y, w], T.grad(loss_(x, y, w), param))

    def predict_proba(self, X):
        result = numpy.zeros([len(X), 2])
        result[:, 1] = self.Activation(X.transpose())
        result[:, 0] = 1 - result[:, 1]
        return result

    def compute_loss(self, X, y, sample_weight=None):
        sample_weight = utils.check_sample_weight(y, sample_weight=sample_weight)
        sample_weight /= numpy.mean(sample_weight)
        return self.Loss(X.transpose(), y, sample_weight)

    def fit(self, X, y, sample_weight=None, stages=None, batch=None, learning_rate=None, penalty=0.0001):
        if not self.prepared:
            self._prepare()
            self.prepared = True
        sample_weight = utils.check_sample_weight(y, sample_weight=sample_weight)
        sample_weight /= numpy.mean(sample_weight)

        if stages is None:
            stages = self.stages
        if batch is None:
            batch = self.batch
        if learning_rate is None:
            learning_rate = self.learning_rate

        for stage in range(stages):
            indices = numpy.random.randint(0, len(X), batch)
            Xp = X[indices, :].transpose()
            yp = y[indices]
            wp = sample_weight[indices]
            for name in self.parameters:
                der = self.derivatives[name](Xp, yp, wp)
                val = self.parameters[name].get_value() * (1. - learning_rate * penalty) - learning_rate * der
                self.parameters[name].set_value(val)


class SimpleNeuralNetwork(AbstractNeuralNetwork):
    def prepare(self):
        assert len(self.layers) == 3 and self.layers[2] == 1
        n1, n2, n3 = self.layers
        # self.Dropper = theano.shared(value=numpy.random.ones(size=n2).astype(theano.config.floatX),
        # name='D', borrow=True)
        W1 = theano.shared(value=numpy.random.normal(size=[n2, n1]).astype(floatX), name='W1')
        W2 = theano.shared(value=numpy.random.normal(size=[n3, n2]).astype(floatX), name='W2')
        self.parameters = {'W1': W1, 'W2': W2}

        def activation(input):
            first = T.nnet.sigmoid(T.dot(W1, input))
            return T.nnet.sigmoid(T.dot(W2, first))
        return activation


class MultiLayerNetwork(AbstractNeuralNetwork):
    def prepare(self):
        activations = [lambda input: input]
        for i, layer in list(enumerate(self.layers))[1:]:
            W = theano.shared(value=numpy.random.normal(size=[self.layers[i], self.layers[i-1]]), name='W' + str(i))
            self.parameters[i] = W
            activations.append(lambda input, i=i: T.nnet.sigmoid(T.dot(self.parameters[i], activations[i - 1](input))))
        return activations[-1]