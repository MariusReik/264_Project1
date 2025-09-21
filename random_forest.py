import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, criterion="entropy", max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)


    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        self.trees = []
        for _ in range(self.n_estimators):
            # bootstrap-trekk
            idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            # tren et tre
            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=int(rng.integers(1e9))  # hold på samme range
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)


    def predict(self, X):
        # samle prediksjoner fra alle trær
        all_preds = [tree.predict(X) for tree in self.trees]

        final = []
        for i in range(len(X)):
            votes = [pred[i] for pred in all_preds]  # stemmer fra alle trær
            vals, counts = np.unique(votes, return_counts=True)
            final.append(vals[np.argmax(counts)])    # flertall
        return np.array(final)



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
