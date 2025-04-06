import numpy as np


class LinearRegressionModel:
    def __init__(
        self,
        learning_rate=0.01,
        iterations=1000,
        fit_intercept=True,
        regularization=None,
        lambda_param=0.5,
    ):
        # Initialize model parameters
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.theta = None  # Model weights

    def _add_intercept(self, X):

        # Add intercept column if required

        return np.column_stack((np.ones(X.shape[0]), X)) if self.fit_intercept else X

    def fit(self, X, y):
        # Convert inputs to NumPy arrays and flatten target variable
        X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64).ravel()

        # Check for NaN or Inf in input data
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Input X contains NaN or Inf values.")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("Input y contains NaN or Inf values.")

        X = self._add_intercept(X)  # Add intercept term if needed
        m, n = X.shape
        self.theta = np.zeros(n)  # Initialize weights to zero

        # Perform gradient descent optimization
        for _ in range(self.iterations):
            gradients = (X.T @ (X @ self.theta - y)) / m  # Compute gradient

            # Apply regularization if specified
            if self.regularization == "l2":
                gradients[1:] += (self.lambda_param / m) * self.theta[
                    1:
                ]  # L2 (Ridge) regularization
            elif self.regularization == "l1":
                gradients[1:] += (self.lambda_param / m) * np.sign(
                    self.theta[1:]
                )  # L1 (Lasso) regularization

            # Update model parameters
            self.theta -= self.learning_rate * gradients

            # Check for NaN or Inf in theta after update
            if np.isnan(self.theta).any() or np.isinf(self.theta).any():
                raise ValueError("Theta contains NaN or Inf values during training.")

    def predict(self, X):
        # Convert to numpy array and add intercept if needed
        X = np.array(X, dtype=np.float64)
        X = self._add_intercept(X)

        # Check for NaN or Inf in input
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("Input X contains NaN or Inf values.")

        predictions = X @ self.theta

        # Check for NaN or Inf in predictions
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            raise ValueError("Predictions contain NaN or Inf values.")

        return predictions
