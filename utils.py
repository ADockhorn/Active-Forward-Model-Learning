import numpy as np
import random


def random_sampling(classifier, X, n_instances: int = 1):
    """
    Random sampling query strategy. Selects a random sample of the training set

    Args:
        classifier: The classifier for which the labels are to be queried.
        X: The pool of samples to query from.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    random_sample = np.array(random.sample(range(X.shape[0]), n_instances))
    return random_sample, X[random_sample]
