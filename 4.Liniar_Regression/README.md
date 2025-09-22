# 📌 Custom Linear Regression

## 📖 Overview
This homework implements a **Custom Linear Regression Model** from scratch using **NumPy and Pandas**. The model supports multiple features, feature scaling, and optional regularization techniques (L1 & L2). It is designed to provide an intuitive and structured approach to training and making predictions with linear regression.

---

## ⚡ Features
- **Supports Multiple Features**: Works with datasets having multiple independent variables.
- **Regularization**: Supports **L1 (Lasso)** and **L2 (Ridge)** regularization.
- **Gradient Descent Optimization**: Implements gradient descent to optimize weights.
- **Intercept Handling**: Option to include/exclude an intercept.



## 🚀 Usage
### Import the Module
```python
from custom_linear_regression import LinearRegressionModel
```

### Initialize the Model
```python
model = LinearRegressionModel(learning_rate=0.01, iterations=1000, fit_intercept=True, regularization='l2', lambda_param=1.0)
```

### Train the Model
```python
model.fit(X_train, y_train)
```

### Make Predictions
```python
predictions = model.predict(X_test)
```

---

## 🛠️ Model Architecture
### **Class: LinearRegressionModel**
This is the core class for implementing linear regression.

#### **Constructor Parameters**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | `0.01` | Step size for gradient descent |
| `iterations` | int | `1000` | Number of training iterations |
| `fit_intercept` | bool | `True` | Whether to include an intercept |
| `regularization` | str | `None` | Type of regularization (`'l1'`, `'l2'`, or `None`) |
| `lambda_param` | float | `1.0` | Regularization strength |

#### **Key Methods**
| Method | Description |
|--------|-------------|
| `fit(X, y)` | Trains the model using gradient descent , computes the cost function (with regularization)|
| `predict(X)` | Predicts target values based on trained weights |
| `_add_intercept(X)` | Adds an intercept column to the feature matrix |

---

## 📊 Mathematical Formulation

### 1. The **gradient descent update rule** is:

$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum (h(\theta) - y) X_j
\
$$

where:
- $ \alpha $ is the learning rate

### 2. **Regularization**

Regularization is applied to prevent overfitting by adding a penalty term to the cost function. The regularized cost function for L2 regularization (Ridge) is:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

For L1 regularization (Lasso), the penalty term is:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|
$$

Where:
- $\lambda$ is the regularization parameter.
- $n$ is the number of features.
- $\theta_j$ is the weight for feature $j$.

### 3. **R² Score (Coefficient of Determination)**

The R² score is a measure of how well the model's predictions match the actual data. It is calculated as:

$$
R^2 = 1 - \frac{\sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m} (y^{(i)} - \bar{y})^2}
$$

Where:
- $y^{(i)}$ is the actual value for the $i$-th data point.
- $\hat{y}^{(i)}$ is the predicted value for the $i$-th data point.
- $\bar{y}$ is the mean of the actual values.


