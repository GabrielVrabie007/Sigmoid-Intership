# Logistic Regression

## Overview

Logistic Regression is a fundamental supervised machine learning algorithm used primarily for binary classification problems. 

## Mathematical Foundation

### The Logistic Function

At the core of logistic regression is the sigmoid function (also called the logistic function), which maps any real-valued number to a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $\sigma(z)$ is the sigmoid function
- $z$ is the input to the function (in our case, the linear combination of features and weights)
- $e$ is the base of the natural logarithm

In the provided code, this is implemented as:

```python
def _sigmoid(self, z):
    z = np.clip(z, -500, 500)  # clip prevent overflow
    return 1 / (1 + np.exp(-z))
```

### Linear Predictor

Before applying the sigmoid function, we compute a linear combination of the input features and the model weights (parameters):

This can be written more compactly in vector form:

$$z = w_0 + \mathbf{w}^T\mathbf{x}$$

Where:
- $w_0$ is the bias term (intercept)
- $\mathbf{w}$ is the weight vector
- $\mathbf{x}$ is the feature vector

In matrix form for multiple samples, this becomes:

$$\mathbf{z} = \mathbf{X}\mathbf{w}$$

In our code, this is calculated during prediction:

```python
z = np.dot(X, self.weights)
predictions = self._sigmoid(z)
```

### Probability Model

The output of the sigmoid function gives us the probability that the input belongs to the positive class:

$$P(y=1|\mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-(w_0 + \mathbf{w}^T\mathbf{x})}}$$

And consequently:

$$P(y=0|\mathbf{x}) = 1 - P(y=1|\mathbf{x}) = 1 - \sigma(z)$$

### Decision Boundary

A decision boundary is established typically at $P(y=1|\mathbf{x}) = 0.5$, which corresponds to $z = 0$. This means:

- If $z > 0$ (or $P(y=1|\mathbf{x}) > 0.5$), classify as class 1
- If $z < 0$ (or $P(y=1|\mathbf{x}) < 0.5$), classify as class 0

In our implementation, a threshold parameter (default 0.5) determines this boundary:

```python
def predict(self, X, threshold=0.5):
    return (self.predict_proba(X) >= threshold).astype(int)
```

## Cost Function

To train the logistic regression model, we need to define a cost function that measures how well our current weights are performing. The cost function used is the binary cross-entropy (log loss):

$$J(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

Where:
- $m$ is the number of training examples
- $y^{(i)}$ is the actual class of the $i$-th example
- $\hat{y}^{(i)} = \sigma(z^{(i)})$ is the predicted probability for the $i$-th example

In our code, this is implemented as:

```python
loss = -np.mean(
    y * np.log(predictions + 1e-15)
    + (1 - y) * np.log(1 - predictions + 1e-15)
)
```

Note the small epsilon value (`1e-15`) added to prevent taking the logarithm of zero.


## Implementation Details

The provided implementation includes:

1. **Initialization**: Setting hyperparameters like learning rate and number of iterations
2. **Fitting**: Training the model using gradient descent
3. **Prediction**: Making class predictions based on learned parameters
4. **Evaluation**: Calculating the accuracy of predictions

The algorithm also handles:
- Adding an intercept term (bias) to the feature matrix
- Tracking the loss history during training
- Preventing numerical overflow in the sigmoid calculation


## Conclusion

Logistic regression is a powerful yet interpretable classification algorithm with solid mathematical foundations in statistics and probability theory. Despite its simplicity, it performs well on many real-world problems, especially when the relationship between features and target classes is approximately linear.