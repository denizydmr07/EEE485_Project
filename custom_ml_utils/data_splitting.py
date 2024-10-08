import numpy as np

np.random.seed(42)

# Train test split
def train_test_split(X, y, test_size=0.2, shuffle=True, random_seed=None):
    if shuffle:
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(X))
        X = X[shuffled_indices]
        y = y[shuffled_indices]

    split_index = len(X) - int(len(X) // (1 / test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test

# Cross validation score
def cross_val_score(model, X, y, cv=5):
    n = X.shape[0]
    indices = np.random.permutation(n)
    X = X[indices]
    y = y[indices]
    scores = np.zeros(cv)

    for i in range(cv):
        split = int(n * (i / cv))
        X_train = np.concatenate([X[:split], X[split + int(n / cv):]])
        X_test = X[split:split + int(n / cv)]
        y_train = np.concatenate([y[:split], y[split + int(n / cv):]])
        y_test = y[split:split + int(n / cv)]

        model.fit(X_train, y_train)
        scores[i] = model.score(X_test, y_test)

    return scores
