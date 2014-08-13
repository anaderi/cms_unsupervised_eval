import numpy
from sklearn.svm import SVC

def check_sample_weight(y_true, sample_weight):
    """Checks the weights, returns normalized version """
    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The lengths are different: {0} and {1}".format(len(y_true), len(sample_weight))
        return sample_weight


class FastSVC(SVC):
    def __init__(self, max_class_samples=200, **kw_args):
        SVC.__init__(self, **kw_args)
        self.max_class_samples = max_class_samples

    def fit(self, X, y, sample_weight=None):
        indices_parts = []
        for label in numpy.unique(y):
            mask = y == label
            indices = numpy.where(mask)[0]
            indices_parts.append(numpy.shuffle(indices)[:self.max_class_samples])

        selected_indices = numpy.sort(numpy.concatenate(indices_parts))

        X = X[selected_indices, :]
        y = y[selected_indices]
        sample_weight = check_sample_weight(y, sample_weight=sample_weight)
        sample_weight = sample_weight[selected_indices]
        self.fit(X, y, sample_weight=sample_weight)