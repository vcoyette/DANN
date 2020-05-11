"""Define domain adaptation tasks.

Available tasks are:
    0. MNIST to MNIST_M
    1. SVHN to MNIST

Each task is represented by a namedtuple containing utils to
load source and target domains and networks definitions.
"""
from collections import namedtuple

from data.utils import load_mnist, load_mnist_m, load_svhn
from models import mnist, svhn


# Each task is represented by the following named tuple:
Task = namedtuple('Task', [
    'load_source',
    'load_target',
    'feature_extractor',
    'classifier',
    'domain_regressor'
])

# Init list of tasks
tasks = []

# MNIST to MNIST_M definition:
task = Task(
    load_source=load_mnist,
    load_target=load_mnist_m,
    feature_extractor=mnist.FeatureExtractor,
    classifier=mnist.Classifier,
    domain_regressor=mnist.DomainRegressor
)

tasks.append(task)

# SVHN to MNIST definition:
task = Task(
    load_source=load_svhn,
    load_target=load_mnist,
    feature_extractor=svhn.FeatureExtractor,
    classifier=svhn.Classifier,
    domain_regressor=svhn.DomainRegressor
)

tasks.append(task)
