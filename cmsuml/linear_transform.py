from __future__ import division, print_function
import numpy
import pylab
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_arrays

__author__ = 'Alex Rogozhnikov'


def exp_potential(r):
    return numpy.exp(-r)


def power_potential(power=-1, lower=0.02):
    def _potential(r):
        return numpy.clip(r, lower, 1e10) ** power
    return _potential


def plot_data(X, y):
    for label in set(y):
        pylab.plot(X[y == label, 0], X[y == label, 1], '.')
    pylab.show()


def generate_linear_transform(X, y, potential=exp_potential,
                              iterations=10, n_neighbors=10, r_neighbours=2., eps=0.05,
                              max_class_data_on_iteration=200,
                              penalize_same_class=True, balance_forces=True):
    """
    This function should return some linear transform, that enlarges the distances between classes
    :return:
    """
    X, y = check_arrays(X, y)
    n_dimensions = X.shape[1]
    y = numpy.array(y, dtype=int)
    labels = set(y)
    X_init = numpy.copy(X)

    resulting_transform = numpy.identity(n_dimensions)

    for iteration in range(iterations):
        nn = NearestNeighbors(n_neighbors=n_neighbors, radius=r_neighbours)
        indices = []
        for label in labels:
            ind = numpy.where(y == label)[0]
            numpy.random.shuffle(ind)
            indices.append(ind[:max_class_data_on_iteration])

        indices = numpy.concatenate(indices)
        assert len(indices) == len(set(indices))
        X_iter = X[indices, :]
        y_iter = y[indices]

        nn.fit(X_iter)
        knn_indices = nn.kneighbors(X_iter, n_neighbors=n_neighbors, return_distance=False)
        knn_indices = knn_indices[:, 1:]

        # in each row two close events
        neighbors = numpy.zeros([knn_indices.shape[0] * knn_indices.shape[1], 2], dtype=numpy.int)
        i = 0
        for index1, knn_1 in zip(range(len(X_iter)), knn_indices):
            for index2 in knn_1:
                neighbors[i] = [index1, index2]
                i += 1

        # each row is vector x_i - x_j, where i and j are neighbouring events
        pairwise_vectors = numpy.zeros([len(neighbors), X.shape[1]])
        same_class = numpy.zeros(len(neighbors), dtype=numpy.int)

        for i, (index1, index2) in enumerate(neighbors):
            same_class[i] = y_iter[index1] == y_iter[index2]
            pairwise_vectors[i] = X_iter[index1, :] - X_iter[index2, :]

        # positive for different classes
        same_class_multiplier = 1 - same_class
        if penalize_same_class:
            same_class_multiplier = 1 - 2 * same_class

        r = numpy.linalg.norm(pairwise_vectors, axis=1)
        assert len(r) == len(pairwise_vectors)

        multiplier = same_class_multiplier * potential(r)

        if balance_forces:
            multiplier[multiplier > 0] /= numpy.sum(multiplier[multiplier > 0])
            multiplier[multiplier < 0] /= - numpy.sum(multiplier[multiplier < 0])

        derivatives = numpy.dot(pairwise_vectors.T,  multiplier[:, numpy.newaxis] * pairwise_vectors)
        derivatives /= numpy.linalg.norm(derivatives, ord='fro')

        # print numpy.linalg.norm(derivatives, ord='fro')

        transformer = numpy.identity(n_dimensions, dtype=numpy.float64) + derivatives * eps
        transformer *= numpy.linalg.det(transformer) ** (- 1. / n_dimensions)

        assert numpy.all(numpy.isfinite(transformer))
        assert 0.99 < numpy.linalg.det(transformer) < 1.01

        resulting_transform = numpy.dot(resulting_transform, transformer)
        X = numpy.dot(X, transformer)
        assert numpy.allclose(X, numpy.dot(X_init, resulting_transform))
        yield resulting_transform


def sandwich_data(num_classes=6, num_points=22, dist=0.4):
    # memory pre-allocation
    x = numpy.zeros([num_classes * num_points, 2])
    y = numpy.zeros(num_classes * num_points, dtype=int)
    for i in xrange(num_classes):
        y[i * num_points: i * num_points + num_points] = i
        for j in xrange(num_points):
            x[i * num_points + j, :] = numpy.array([numpy.random.normal(j, 0.1), numpy.random.normal(dist * i, 0.1)])
    return x, y


def gauss_data(n_samples=1000, n_features=10, classes=5, std=1.):
    from sklearn.datasets import make_blobs
    return make_blobs(n_samples=n_samples, n_features=n_features, cluster_std=std, centers=classes)











