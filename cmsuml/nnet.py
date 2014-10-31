"""
This is simple version of neuralNetworks, which can be easily adopted.
The neural networks from this library provide sklearn classifier's interface.
"""
from __future__ import print_function, division

import numpy
from sklearn.utils.validation import check_random_state
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator, ClassifierMixin

from cmsuml import utils


floatX = theano.config.floatX
__author__ = 'Alex Rogozhnikov'


#region Loss functions

def squared_loss(y, pred, w):
    return T.mean(w * (y - pred) ** 2)


def log_loss(y, pred, w):
    return -T.mean(w * (y * T.log(pred) + (1 - y) * T.log(1 - pred)))


def ada_loss(y, pred, w):
    """important - ada loss should be used with nnets without sigmoid,
    output should be arbitrary real, not [0,1]"""
    return T.mean(w * T.exp(pred * (1 - 2. * y)))

#endregion


#region Trainers
def get_batch(xT, y, w, batch=10, random=numpy.random):
    """ Generates subset of training dataset, of size batch"""
    if len(y) > batch:
        indices = random.choice(len(xT), size=batch)
        return xT[:, indices], y[indices], w[indices]
    else:
        return xT, y, w


def sgd_trainer(x, y, w, parameters, derivatives, loss,
                stages=1000, batch=10, learning_rate=0.1, l2_penalty=0.001, random=numpy.random):
    """Simple gradient descent with backpropagation"""
    xT = x.T
    for stage in range(stages):
        xTp, yp, wp = get_batch(xT, y, w, batch=batch, random=random)
        for name in parameters:
            der = derivatives[name](xTp, yp, wp)
            val = parameters[name].get_value() * (1. - learning_rate * l2_penalty) - learning_rate * der
            parameters[name].set_value(val)


def irprop_minus_trainer(x, y, w, parameters, derivatives, loss, stages=100, max_stage_samples=1000,
                         positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6, random=numpy.random):
    """ IRPROP- trainer, see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428 """
    deltas = dict([(name, 1e-3 * numpy.ones_like(p)) for name, p in parameters.iteritems()])
    prev_derivatives = dict([(name, numpy.zeros_like(p)) for name, p in parameters.iteritems()])
    xT = x.T
    for _ in range(stages):
        xTp, yp, wp = get_batch(xT, y, w, batch=max_stage_samples, random=random)
        for name in parameters:
            new_derivative = derivatives[name](xTp, yp, wp)
            old_derivative = prev_derivatives[name]
            delta = deltas[name]
            delta = numpy.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
            delta = numpy.clip(delta, min_step, max_step)
            deltas[name] = delta
            val = parameters[name].get_value()
            parameters[name].set_value(val - delta * numpy.sign(new_derivative))
            new_derivative[new_derivative * old_derivative < 0] = 0
            prev_derivatives[name] = new_derivative


def irprop_plus_trainer(x, y, w, parameters, derivatives, loss, stages=100, max_stage_samples=1000,
                        positive_step=1.2, negative_step=0.5, max_step=1., min_step=1e-6, random=numpy.random):
    """IRPROP+ trainer, see http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332"""
    deltas = dict([(name, 1e-3 * numpy.ones_like(p)) for name, p in parameters.iteritems()])
    prev_derivatives = dict([(name, numpy.zeros_like(p)) for name, p in parameters.iteritems()])
    prev_loss_value = 1e10
    xT = x.T
    for _ in range(stages):
        xTp, yp, wp = get_batch(xT, y, w, batch=max_stage_samples, random=random)
        loss_value = loss(xTp, yp, wp)
        for name in parameters:
            new_derivative = derivatives[name](xTp, yp, wp)
            old_derivative = prev_derivatives[name]
            val = parameters[name].get_value()
            delta = deltas[name]
            if loss_value > prev_loss_value:
                # step back
                val += numpy.where(new_derivative * old_derivative < 0, delta * numpy.sign(old_derivative), 0)
            delta = numpy.where(new_derivative * old_derivative > 0, delta * positive_step, delta * negative_step)
            delta = numpy.clip(delta, min_step, max_step)
            deltas[name] = delta
            val -= numpy.where(new_derivative * old_derivative >= 0, delta * numpy.sign(new_derivative), 0)
            parameters[name].set_value(val)
            new_derivative[new_derivative * old_derivative < 0] = 0
            prev_derivatives[name] = new_derivative
        prev_loss_value = loss_value


trainers = {'sgd': sgd_trainer, 'irprop-': irprop_minus_trainer, 'irprop+': irprop_plus_trainer}
#endregion


# TODO think of dropper and noises (and maybe something else)
# TODO add random state
class AbstractNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, layers=None, loss=log_loss, trainer='irprop-', trainer_parameters=None, random_state=None):
        """
        Constructs the neural network based on Theano (for classification purposes).
        Works in sklearn fit-predict way: X is [n_samples, n_features], y is [n_samples], sample_weight is [n_samples].
        Works as usual sklearn classifier, can be used in boosting, for instance, pickled, etc.
        :param layers: list of int, e.g [14, 7, 1] - the number of units in each layer
        :param loss: loss function used (log_loss by default)
        :param trainer: string, describes the method
        :param trainer_parameters: parameters passed to trainer function (learning_rate, etc., trainer-specific).
        """
        self.layers = layers
        self.loss = loss
        self.prepared = False
        self.parameters = {}
        self.derivatives = {}
        self.trainer = trainer
        self.trainer_parameters = trainer_parameters
        self.random_state = random_state
        self.classes_ = numpy.array([0, 1])  # Dirty hack for AdaBoost

    def prepare(self):
        """This method should provide activation function and set parameters
        :return Activation function, f: X.T -> p,
        X.T of shape [n_features, n_events], p of shape [n_events, n_outputs]
        """
        raise NotImplementedError()

    def _prepare(self):
        """This function is called once, it creates the activation function, it's gradient
        and initializes the weights"""
        self.random_state = check_random_state(self.random_state)
        activation = self.prepare()
        loss_ = lambda x, y, w: self.loss(y, activation(x), w)
        # TODO maybe use matrices here?
        x = T.matrix('X')
        y = T.vector('y')
        w = T.vector('w')
        self.Activation = theano.function([x], activation(x))
        self.Loss = theano.function([x, y, w], loss_(x, y, w))
        for name, param in self.parameters.iteritems():
            self.derivatives[name] = theano.function([x, y, w], T.grad(loss_(x, y, w), param))

    def activate(self, X):
        """ Activates NN on particular dataset
        :param numpy.array X: of shape [n_samples, n_features] """
        return self.Activation(numpy.transpose(X))

    def predict_proba(self, X):
        """Computes praobability of each event to belong to a particular class
        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples, n_classes]
        """
        result = numpy.zeros([len(X), 2])
        result[:, 1] = self.Activation(X.transpose())
        result[:, 0] = 1 - result[:, 1]
        return result

    def predict(self, X):
        """ Predicted
        :param numpy.array X: of shape [n_samples, n_features]
        :return: numpy.array of shape [n_samples] with labels of predicted classes """
        return self.predict_proba(X).argmax(axis=1)

    def compute_loss(self, X, y, sample_weight=None):
        """Computes chosen loss on labeled dataset
        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array with integer labels of shape [n_samples],
            in two-class classification 0 and 1 labels should be used
        :param sample_weight: optional, numpy.array of shape [n_samples],
            weights are normalized (so that 'mean' == 1).
        :return float, the loss vales computed"""
        sample_weight = utils.check_sample_weight(y, sample_weight=sample_weight)
        sample_weight /= numpy.mean(sample_weight)
        return self.Loss(X.transpose(), y, sample_weight)

    def fit(self, X, y, sample_weight=None, trainer=None, **trainer_parameters):
        """ Prepare the model by optimizing selected loss function with some trainer.
        This method can (and should) be called several times, each time with new parameters
        :param X: numpy.array of shape [n_samples, n_features]
        :param y: numpy.array of shape [n_samples]
        :param sample_weight: numpy.array of shape [n_samples], leave None for array of 1's
        :param trainer: str, method used to minimize loss, overrides one in the ctor
        :param trainer_parameters: parameters for this method, override ones in ctor
        :return: self """
        if not self.prepared:
            self._prepare()
            self.prepared = True
        sample_weight = utils.check_sample_weight(y, sample_weight=sample_weight)
        sample_weight /= numpy.mean(sample_weight)

        trainer = trainers[self.trainer if trainer is None else trainer]
        parameters_ = {} if self.trainer_parameters is None else self.trainer_parameters.copy()
        parameters_.update(trainer_parameters)
        trainer(X, y, sample_weight, self.parameters, self.derivatives, self.Loss,  **parameters_)
        return self


#region Neural networks

class SimpleNeuralNetwork(AbstractNeuralNetworkClassifier):
    """The most simple NN with one hidden layer (sigmoid activation) """
    def prepare(self):
        n1, n2, n3 = self.layers
        W1 = theano.shared(value=numpy.random.normal(size=[n2, n1]).astype(floatX), name='W1')
        W2 = theano.shared(value=numpy.random.normal(size=[n3, n2]).astype(floatX), name='W2')
        self.parameters = {'W1': W1, 'W2': W2}

        def activation(input):
            first = T.nnet.sigmoid(T.dot(W1, input))
            return T.nnet.sigmoid(T.dot(W2, first))
        return activation


class MultiLayerNetwork(AbstractNeuralNetworkClassifier):
    """Supports arbitrary number of layers (sigmoid activation each)."""
    def prepare(self):
        activations = [lambda x: x]
        for i, layer in list(enumerate(self.layers))[1:]:
            W = theano.shared(value=numpy.random.normal(size=[self.layers[i], self.layers[i-1]]), name='W' + str(i))
            self.parameters[i] = W
            # j = i trick is to avoid lambda-capturing of i
            activations.append(lambda x, j=i: T.nnet.sigmoid(T.dot(self.parameters[j], activations[j - 1](x))))
        return activations[-1]


class RBFNeuralNetwork(AbstractNeuralNetworkClassifier):
    """One hidden layer with normalized RBF activation (Radial Basis Function)"""
    def prepare(self):
        n1, n2, n3 = self.layers
        W1 = theano.shared(value=numpy.random.normal(size=[n2, n1]).astype(floatX), name='W1')
        W2 = theano.shared(value=numpy.random.normal(size=[n3, n2]).astype(floatX), name='W2')
        G  = theano.shared(value=0.1, name='G')
        self.parameters = {'W1': W1, 'W2': W2, 'G': G}

        def activation(input):
            translation_vectors = W1.reshape((W1.shape[0], 1, -1)) - input.transpose().reshape((1, input.shape[1], -1))
            minkowski_distances = (abs(translation_vectors) ** 2).sum(2)
            first = T.nnet.softmax(- (0.001 + G * G) * minkowski_distances)
            return T.nnet.sigmoid(T.dot(W2, first))
        return activation


class SoftmaxNeuralNetwork(AbstractNeuralNetworkClassifier):
    """One hidden layer, softmax activation function """
    def prepare(self):
        n1, n2, n3 = self.layers
        W1 = theano.shared(value=numpy.random.normal(size=[n2, n1]).astype(floatX), name='W1')
        W2 = theano.shared(value=numpy.random.normal(size=[n3, n2]).astype(floatX), name='W2')
        self.parameters = {'W1': W1, 'W2': W2}

        def activation(input):
            first = T.nnet.softmax(T.dot(W1, input))
            return T.nnet.sigmoid(T.dot(W2, first))
        return activation

#endregion
