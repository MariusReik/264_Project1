import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)  # bootstrap sample
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  # shape: (n_estimators, n_samples)
        # majority vote along axis 0
        final_preds = []
        for i in range(X.shape[0]):
            values, counts = np.unique(tree_preds[:, i], return_counts=True)
            final_preds.append(values[np.argmax(counts)])
        return np.array(final_preds)


if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
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

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
