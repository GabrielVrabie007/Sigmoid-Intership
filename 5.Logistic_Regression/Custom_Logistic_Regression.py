import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, fit_intercept=True):
        """
        Initializes the Logistic Regression algorithm

        Parameters:
        learning_rate (float): Learning rate for gradient descent
        num_iterations (int): Number of iterations for gradient descent
        fit_intercept (bool): Whether to include a bias/intercept term
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        """
        Sigmoid function: 1 / (1 + e^(-z))
        """
        # Used to prevent overflow in exp function
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        """
        Adds a column of 1s for intercept/bias
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X, y, verbose=False):
        """
        Trains the model using gradient descent

        Parameters:
        X (numpy.ndarray): Features, shape (n_samples, n_features)
        y (numpy.ndarray): Target labels, shape (n_samples,)
        verbose (bool): Whether to display training progress
        """
        # Add bias if necessary
        if self.fit_intercept:
            X = self._add_intercept(X)

        # Parameters initialization
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.loss_history = []

        # Gradient descent
        for i in range(self.num_iterations):
            # Predict Calculus
            z = np.dot(X, self.weights)
            predictions = self._sigmoid(z)

            # Error Calculus
            error = predictions - y

            # Gradient Calculus
            gradient = np.dot(X.T, error) / n_samples

            # Params Update
            self.weights -= self.learning_rate * gradient

            # Loss Calculus
            loss = -np.mean(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.loss_history.append(loss)

            # Verbose output for visualization progress
            if verbose and i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict_proba(self, X):
        """
        Returns probabilities for the positive class
        """
        if self.fit_intercept:
            X = self._add_intercept(X)

        return self._sigmoid(np.dot(X, self.weights))

    def predict(self, X, threshold=0.5):
        """
        Returns class predictions (0 or 1)

        Parameters:
        X (numpy.ndarray): Features, shape (n_samples, n_features)
        threshold (float): Threshold for decision (default 0.5)
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculates accuracy on the given set
        """
        print(
            "This print is called from custom Logistic Regression class to make sure it is working correctly"
        )
        return np.mean(self.predict(X) == y)
