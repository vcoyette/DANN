"""Define domain adaptation tasks.

Available tasks are:
    0. MNIST to MNIST_M
    1. SVHN to MNIST
    2. AMAZON to WEBCAM
    3. Syn Signs to GTSRB

Each task is represented by a namedtuple containing utils to
load source and target domains and networks definitions.
"""
from collections import namedtuple

from data.utils import load_mnist, load_mnist_m, load_svhn, load_office,\
                       load_gtsrb, load_synsigns
from models import mnist, svhn, office, synsign


# Each task is represented by the following named tuple:
Task = namedtuple('Task', [
    'load_source',
    'load_target',
    'feature_extractor',
    'classifier',
    'domain_regressor',
    'schedule_lr'
])

# Init list of tasks
tasks = []

# MNIST to MNIST_M definition:
task = Task(
    load_source=load_mnist,
    load_target=load_mnist_m,
    feature_extractor=mnist.FeatureExtractor,
    classifier=mnist.Classifier,
    domain_regressor=mnist.DomainRegressor,
    schedule_lr=True
)

tasks.append(task)

# SVHN to MNIST definition:
task = Task(
    load_source=load_svhn,
    load_target=load_mnist,
    feature_extractor=svhn.FeatureExtractor,
    classifier=svhn.Classifier,
    domain_regressor=svhn.DomainRegressor,
    schedule_lr=False
)

tasks.append(task)

# AMAZON to WEBCAM definition
task = Task(
    load_source=lambda **kwargs: load_office('amazon', **kwargs),
    load_target=lambda **kwargs: load_office('webcam', **kwargs),
    feature_extractor=office.FeatureExtractor,
    classifier=office.Classifier,
    domain_regressor=office.DomainRegressor,
    schedule_lr=True
)

tasks.append(task)

# Syn Sign to GTSRB
task = Task(
    load_source=load_synsigns,
    load_target=load_gtsrb,
    feature_extractor=synsign.FeatureExtractor,
    classifier=synsign.Classifier,
    domain_regressor=synsign.DomainRegressor,
    schedule_lr=True
)

tasks.append(task)
