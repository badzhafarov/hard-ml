from typing import Iterable
import numpy as np


class Node:
    def __init__(self, n_samples, delta_delta_p):
        self.n_samples = n_samples
        self.delta_delta_p = delta_delta_p
        self.feature_index = 0
        self.uplift = None
        self.left = None
        self.right = None


def get_thresholds(column_values):
    unique_values = np.unique(column_values)
    if len(unique_values) > 10:
        percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
    else:
        percentiles = np.percentile(unique_values, [10, 50, 90])
    return np.unique(percentiles)


class UpliftTreeRegressor:
    def __init__(
            self,
            max_depth: int = 3,
            min_samples_leaf: int = 6000,
            min_samples_leaf_treated: int = 2500,
            # min_samples_leaf for objects with T=1 in a leaf.
            min_samples_leaf_control: int = 2500,
            # min_samples_leaf for objects with T=0 in a leaf.
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control

    def fit(
            self,
            X: np.ndarray,  # array (n * k) with features.
            treatment: np.ndarray,  # array (n) with treatment flags.
            y: np.ndarray  # array (n) with targets.
    ) -> None:
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, treatment, y)

    def _predict(self, input: np.ndarray) -> float:
        node = self.tree
        while node.left:
            if input[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.uplift

    def predict(self, X: np.ndarray) -> Iterable[float]:
        return np.array([self._predict(inputs) for inputs in X])

    # uplift functions

    def get_delta_delta_p(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray, threshold: float) -> float:
        left_uplift = self.uplift(treatment[X <= threshold], y[X <= threshold])
        right_uplift = self.uplift(treatment[X > threshold], y[X > threshold])
        return np.abs(left_uplift - right_uplift)

    def uplift(self, treatment: np.ndarray, y: np.ndarray) -> float:
        c_sum = np.sum(y[treatment == 1])
        c_count = len(y[treatment == 1])

        t_sum = np.sum(y[treatment == 0])
        t_count = len(y[treatment == 0])

        return (c_sum / c_count) - (t_sum / t_count)

    # splitting functions

    def _split_on_left_right(self, X, treatment, y, threshold):
        return X[X <= threshold], treatment[X <= threshold], y[X <= threshold], \
               X[X > threshold], treatment[X > threshold], y[X > threshold]

    def _check_min_samples(self, treatment: np.ndarray) -> bool:
        if (len(treatment) >= self.min_samples_leaf and
                len(treatment[treatment == 1]) >= self.min_samples_leaf_treated and
                len(treatment[treatment == 0]) >= self.min_samples_leaf_control):
            return True
        return False

    def _best_split(self, X, treatment, y) -> (int, float):
        best_delta_delta_p = 0
        best_threshold, best_idx_feature = None, None
        for idx in range(self.n_features):
            possible_thr = get_thresholds(X[:, idx])

            possible_thr = list(possible_thr)
            for thr in possible_thr:
                left_X, left_treatment, left_y, right_X, right_treatment, right_y = \
                    self._split_on_left_right(X[:, idx], treatment, y, threshold=thr)

                if self._check_min_samples(left_treatment) and self._check_min_samples(right_treatment):
                    delta_delta_p = self.get_delta_delta_p(X[:, idx], treatment, y, threshold=thr)

                    if delta_delta_p > best_delta_delta_p:
                        best_threshold = thr
                        best_idx_feature = idx
                        best_delta_delta_p = delta_delta_p

        return best_idx_feature, best_threshold

    def _grow_tree(self, X, treatment, y, depth: int = 0) -> Node:
        node = Node(n_samples=X.shape[0], delta_delta_p=0)
        node.uplift = self.uplift(treatment, y)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, treatment, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                left_X, left_treatment, left_y = X[indices_left], treatment[indices_left], y[indices_left]
                right_X, right_treatment, right_y = X[~indices_left], treatment[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(left_X, left_treatment, left_y, depth + 1)
                node.right = self._grow_tree(right_X, right_treatment, right_y, depth + 1)
        return node


if __name__ == '__main__':
    path = 'data'
    X = np.load(f'{path}/example_X.npy')
    y = np.load(f'{path}/example_y.npy')
    treatment = np.load(f'{path}/example_treatment.npy')
    regressor = UpliftTreeRegressor()
    regressor.fit(X, treatment, y)
    predicts = regressor.predict(X)