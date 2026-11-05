import numpy as np

class NumpyMLPClassifier:
    """
    Multi-Layer Perceptron Classifier implemented from scratch with NumPy
    Architecture: Input (d) -> Hidden 1 (16, ReLU) -> Hidden 2 (8, ReLU) -> Output (1, Sigmoid)
    Matches the exact architecture and training logic in Assignment 03 notebooks.
    """
    def __init__(self, hidden_dim1=16, hidden_dim2=8, lr=0.05, epochs=60, batch_size=1024, random_state=42):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = {}
        self.loss_history = []
        self.val_loss_history = []

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(np.float32)

    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _bce_loss(self, y_true, y_pred):
        p = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        y = np.asarray(y_true).reshape(-1, 1)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n_samples, d = X.shape

        rng = np.random.default_rng(self.random_state)
        W1 = (rng.standard_normal((d, self.hidden_dim1)) * np.sqrt(2.0 / d)).astype(np.float32)
        b1 = np.zeros((1, self.hidden_dim1), dtype=np.float32)

        W2 = (rng.standard_normal((self.hidden_dim1, self.hidden_dim2)) * np.sqrt(2.0 / self.hidden_dim1)).astype(np.float32)
        b2 = np.zeros((1, self.hidden_dim2), dtype=np.float32)

        W3 = (rng.standard_normal((self.hidden_dim2, 1)) * np.sqrt(2.0 / self.hidden_dim2)).astype(np.float32)
        b3 = np.zeros((1, 1), dtype=np.float32)

        self.loss_history = []
        self.val_loss_history = []

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start:start + self.batch_size]
                X_b = X[idx]
                y_b = y[idx]
                n_b = len(X_b)

                # Forward
                z1 = X_b @ W1 + b1
                h1 = self._relu(z1)
                z2 = h1 @ W2 + b2
                h2 = self._relu(z2)
                z3 = h2 @ W3 + b3
                y_hat = self._sigmoid(z3)

                # Backward
                dz3 = (y_hat - y_b) / n_b
                dW3 = h2.T @ dz3
                db3 = dz3.sum(axis=0, keepdims=True)

                dh2 = dz3 @ W3.T
                dz2 = dh2 * self._relu_derivative(z2)
                dW2 = h1.T @ dz2
                db2 = dz2.sum(axis=0, keepdims=True)

                dh1 = dz2 @ W2.T
                dz1 = dh1 * self._relu_derivative(z1)
                dW1 = X_b.T @ dz1
                db1 = dz1.sum(axis=0, keepdims=True)

                # Update
                W1 -= self.lr * dW1
                b1 -= self.lr * db1
                W2 -= self.lr * dW2
                b2 -= self.lr * db2
                W3 -= self.lr * dW3
                b3 -= self.lr * db3

            # Record loss
            train_pred = self._forward_pass(X, W1, b1, W2, b2, W3, b3)
            self.loss_history.append(self._bce_loss(y, train_pred))
            if X_val is not None and y_val is not None:
                val_pred = self._forward_pass(X_val, W1, b1, W2, b2, W3, b3)
                self.val_loss_history.append(self._bce_loss(y_val, val_pred))

        self.weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        return self

    def _forward_pass(self, X, W1, b1, W2, b2, W3, b3):
        X = np.asarray(X, dtype=np.float32)
        h1 = self._relu(X @ W1 + b1)
        h2 = self._relu(h1 @ W2 + b2)
        return self._sigmoid(h2 @ W3 + b3)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        p1 = self._forward_pass(
            X,
            self.weights["W1"], self.weights["b1"],
            self.weights["W2"], self.weights["b2"],
            self.weights["W3"], self.weights["b3"]
        ).ravel()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


class NumpyMLPRegressor:
    """
    Multi-Layer Perceptron Regressor implemented from scratch with NumPy
    Architecture: Input (d) -> Hidden 1 (16, ReLU) -> Hidden 2 (8, ReLU) -> Output (1, Linear)
    Matches the exact architecture and training logic in House Price notebook.
    """
    def __init__(self, hidden_dim1=16, hidden_dim2=8, lr=0.01, epochs=60, batch_size=1024, random_state=42):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = {}
        self.y_mean = 0.0
        self.y_std = 1.0
        self.loss_history = []
        self.val_loss_history = []

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return (z > 0).astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n_samples, d = X.shape

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) if np.std(y) > 0 else 1.0

        y_norm = ((y - self.y_mean) / self.y_std).astype(np.float32).reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)
        W1 = (rng.standard_normal((d, self.hidden_dim1)) * np.sqrt(2.0 / d)).astype(np.float32)
        b1 = np.zeros((1, self.hidden_dim1), dtype=np.float32)

        W2 = (rng.standard_normal((self.hidden_dim1, self.hidden_dim2)) * np.sqrt(2.0 / self.hidden_dim1)).astype(np.float32)
        b2 = np.zeros((1, self.hidden_dim2), dtype=np.float32)

        W3 = (rng.standard_normal((self.hidden_dim2, 1)) * np.sqrt(2.0 / self.hidden_dim2)).astype(np.float32)
        b3 = np.zeros((1, 1), dtype=np.float32)

        self.loss_history = []
        self.val_loss_history = []

        for epoch in range(self.epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start:start + self.batch_size]
                X_b = X[idx]
                y_b = y_norm[idx]
                n_b = len(X_b)

                # Forward
                z1 = X_b @ W1 + b1
                h1 = self._relu(z1)
                z2 = h1 @ W2 + b2
                h2 = self._relu(z2)
                y_hat = h2 @ W3 + b3

                # Backward (MSE derivative: 2 * (y_hat - y) / n)
                dz3 = 2.0 * (y_hat - y_b) / n_b
                dW3 = h2.T @ dz3
                db3 = dz3.sum(axis=0, keepdims=True)

                dh2 = dz3 @ W3.T
                dz2 = dh2 * self._relu_derivative(z2)
                dW2 = h1.T @ dz2
                db2 = dz2.sum(axis=0, keepdims=True)

                dh1 = dz2 @ W2.T
                dz1 = dh1 * self._relu_derivative(z1)
                dW1 = X_b.T @ dz1
                db1 = dz1.sum(axis=0, keepdims=True)

                # Update
                W1 -= self.lr * dW1
                b1 -= self.lr * db1
                W2 -= self.lr * dW2
                b2 -= self.lr * db2
                W3 -= self.lr * dW3
                b3 -= self.lr * db3

            train_hat = self._forward_pass(X, W1, b1, W2, b2, W3, b3)
            self.loss_history.append(float(np.mean((y_norm - train_hat) ** 2)))
            if X_val is not None and y_val is not None:
                val_norm = ((y_val - self.y_mean) / self.y_std).astype(np.float32).reshape(-1, 1)
                val_hat = self._forward_pass(X_val, W1, b1, W2, b2, W3, b3)
                self.val_loss_history.append(float(np.mean((val_norm - val_hat) ** 2)))

        self.weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
        return self

    def _forward_pass(self, X, W1, b1, W2, b2, W3, b3):
        X = np.asarray(X, dtype=np.float32)
        h1 = self._relu(X @ W1 + b1)
        h2 = self._relu(h1 @ W2 + b2)
        return h2 @ W3 + b3

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        y_hat_norm = self._forward_pass(
            X,
            self.weights["W1"], self.weights["b1"],
            self.weights["W2"], self.weights["b2"],
            self.weights["W3"], self.weights["b3"]
        ).ravel()
        return y_hat_norm * self.y_std + self.y_mean

