import faiss
import warnings

from utils.constants import *


class DataStore(object):
    """
    Builds a Datastore object, i.e, the set of all key-value pairs constructed from the training examples.
    One training example, consists in a vector (of size `dimension`) and a label (`int`).
    The datastore has a maximal number of examples that it can store (`capacity`).

    Attributes
    ----------
    capacity:
    strategy:
    dimension:
    rng (numpy.random._generator.Generator):
    index:
    labels:

    Methods
    -------
    __init__
    build
    clear

    """
    def __init__(self, capacity, strategy, dimension, rng):
        self.capacity = capacity
        self.strategy = strategy
        self.dimension = dimension
        self.rng = rng

        self.index = faiss.IndexFlatL2(self.dimension)
        self.labels = None

    @property
    def strategy(self):
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy):
        if strategy in ALL_STRATEGIES:
            self.__strategy = strategy
        else:
            warnings.warn("strategy is set to random!", RuntimeWarning)
            self.__strategy = "random"

    def build(self, train_vectors, train_labels):
        """
        add vectors to `index` according to `strategy`

        :param train_vectors:
        :type train_vectors: numpy.array os shape (n_samples, dimension)
        :param train_labels:
        :type train_labels: numpy.array of shape (n_samples,)

        """
        if self.capacity <= 0:
            return

        n_train_samples = len(train_vectors)
        if n_train_samples <= self.capacity:
            self.index.add(train_vectors)
            self.labels = train_labels
            return

        if self.strategy == "random":
            selected_indices = self.rng.choice(n_train_samples, size=self.capacity, replace=False)
            selected_vectors = train_vectors[selected_indices]
            selected_labels = train_labels[selected_indices]

            self.index.add(selected_vectors)
            self.labels = selected_labels
        elif self.strategy == "stratified":
            unique_labels, label_counts = np.unique(train_labels, return_counts=True)

            selected_vectors = []
            selected_labels = []
            total_samples = 0

            for i in range(len(unique_labels)):
                label = unique_labels[i]
                label_features = train_vectors[train_labels == label]
                label_labels = train_labels[train_labels == label]
                label_count = label_counts[i]

                proportion = label_counts[i] / len(train_vectors)

                num_samples_per_label = int(self.capacity * proportion)
                num_samples_per_label = min(num_samples_per_label, label_count)

                indices = np.random.choice(label_count, size=num_samples_per_label, replace=False)
                selected_vectors.extend(label_features[indices])
                selected_labels.extend(label_labels[indices])

            selected_vectors = np.array(selected_vectors)
            selected_labels = np.array(selected_labels)

            if len(selected_vectors) < self.capacity:
                remaining_samples = self.capacity - len(selected_vectors)

                selected_indices = np.arange(total_samples)
                available_indices = np.setdiff1d(np.arange(len(train_vectors)), selected_indices)

                indices = np.random.choice(available_indices, size=remaining_samples, replace=False)
                selected_vectors = np.concatenate((selected_vectors, train_vectors[indices]))
                selected_labels = np.concatenate((selected_labels, train_labels[indices]))

            self.index.add(selected_vectors)
            self.labels = selected_labels
        else:
            raise NotImplementedError(f"{self.strategy} is not implemented")

    def clear(self):
        """
        empties the datastore by reinitializing `index` and clearing  `labels`

        """
        self.index = faiss.IndexFlatL2(self.dimension)
        self.labels = None
