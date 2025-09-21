import numpy as np
from typing import Self
import math

"""
Decision tree implementation INF264 Project 1 Marius og lyder

"""

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    labels, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return probs


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    probs = count(y)
    return 1 - np.sum(probs ** 2)


def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    probs = count(y)
    result = 0
    for p in probs:
        if p>0:
            result += p * math.log2(p)
    
    return -result

def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    return x <= value


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

class Node:
    def __init__(self, feature=0, threshold=0.0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features: None | str = None,
        random_state: int | None = None,
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # velg impurity-funksjon
        def impurity(y):
            if self.criterion == "entropy":
                return entropy(y)
            elif self.criterion == "gini":
                return gini_index(y)
            else:
                raise ValueError("Ukjent kriterie")

        def best_split(X, y):

            best_feature, best_threshold, best_gain = None, None, -np.inf
            n_features = X.shape[1]

            # hvor mange features som vurderes
            if self.max_features == "sqrt":
                n_sub = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_sub = int(np.log2(n_features))
            else:
                n_sub = n_features

            # trekk tilfeldige features
            chosen = self.rng.choice(n_features, n_sub, replace=False)

            current_impurity = impurity(y)

            for feature in chosen:
                values = X[:, feature]
                threshold = np.mean(values)  # gjennomsnitt som threshold
                left_mask = split(values, threshold)
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_impurity = impurity(y[left_mask])
                right_impurity = impurity(y[right_mask])
                left_weight = np.sum(left_mask) / len(y)
                right_weight = np.sum(right_mask) / len(y)
                gain = current_impurity - (left_weight * left_impurity + right_weight * right_impurity)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

            return best_feature, best_threshold, best_gain

        def build_tree(X, y, depth):

            if len(np.unique(y)) == 1:  # alle labels like
                return Node(value=y[0])
            if all(np.array_equal(row, X[0]) for row in X):  # alle rader like
                return Node(value=most_common(y))
            if self.max_depth is not None and depth >= self.max_depth:  # nådd maks dybde
                return Node(value=most_common(y))

            feature, threshold, gain = best_split(X, y)

            if feature is None or gain <= 0:  # ingen nyttig split
                return Node(value=most_common(y))

            left_mask = split(X[:, feature], threshold)
            right_mask = ~left_mask
            left = build_tree(X[left_mask], y[left_mask], depth + 1)
            right = build_tree(X[right_mask], y[right_mask], depth + 1)
            return Node(feature=feature, threshold=threshold, left=left, right=right)

        # bygging av treet
        self.root = build_tree(X, y, 0)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        def traverse(x, node):
            while not node.is_leaf():
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.value
        return np.array([traverse(x, self.root) for x in X])


if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="entropy")
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")