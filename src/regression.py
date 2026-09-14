"""
src/regression.py - Task 4 Core Regression Algorithms

Implemented strictly from first principles using pure NumPy (no scikit-learn,
statsmodels, or scipy allowed).

Contains:
    - hypothesis(X, theta): Compute linear model predictions h_theta(X) = X @ theta
    - cost(X, y, theta): Compute least-squares cost J(theta) = 1/(2n) * sum((X@theta - y)^2)
    - fit_normal(X, y): Closed-form analytical solution via Normal Equation: theta = (X^T X)^(-1) X^T y
    - fit_batch_gd(X, y, alpha, n_iters): Batch Gradient Descent optimizer
    - fit_sgd(X, y, alpha, n_epochs): Stochastic Gradient Descent optimizer
    - rmse(X, y, theta): Root Mean Squared Error evaluation metric with clipping at zero
"""

import numpy as np


def hypothesis(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute hypothesis predictions for feature matrix X and parameters theta.

    Formula:
        h_theta(x) = theta^T x = sum_{j=0}^d theta_j * x_j
        In matrix form: h_theta(X) = X @ theta

    Args:
        X: Design matrix of shape (n, d+1) including the intercept column x_0 = 1
        theta: Parameter vector of shape (d+1,)

    Returns:
        np.ndarray of shape (n,) containing linear predictions
    """
    return np.dot(X, theta)


def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute the least-squares mean squared error cost function J(theta).

    Formula (Lecture 1 / Section 2.4):
        J(theta) = 1 / (2 * n) * sum_{i=1}^n (h_theta(x^(i)) - y^(i))^2

    Args:
        X: Design matrix of shape (n, d+1)
        y: Ground truth target vector of shape (n,)
        theta: Parameter vector of shape (d+1,)

    Returns:
        float scalar cost value J(theta)
    """
    n = len(y)
    if n == 0:
        return 0.0
    predictions = hypothesis(X, theta)
    errors = predictions - y
    return float(np.sum(errors ** 2) / (2.0 * n))


def fit_normal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression using the analytical Normal Equation.

    Formula (Task 4.4):
        theta = (X^T X)^(-1) X^T y

    Args:
        X: Design matrix of shape (n, d+1)
        y: Target vector of shape (n,)

    Returns:
        theta: Optimal parameter vector of shape (d+1,)
    """
    XT_X = np.dot(X.T, X)
    XT_y = np.dot(X.T, y)
    # Using np.linalg.inv as explicitly allowed in the instructions
    theta = np.dot(np.linalg.inv(XT_X), XT_y)
    return theta


def fit_batch_gd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_iters: int,
    initial_theta: np.ndarray = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Fit linear regression using Batch Gradient Descent.

    Formula (Task 4.5):
        Starting from theta = 0, update simultaneously for every j:
            theta_j := theta_j + alpha / n * sum_{i=1}^n (y^(i) - h_theta(x^(i))) * x_j^(i)
        In vector notation:
            theta := theta - (alpha / n) * X^T (X @ theta - y)

    Args:
        X: Design matrix of shape (n, d+1)
        y: Target vector of shape (n,)
        alpha: Learning rate parameter
        n_iters: Total number of iterations
        initial_theta: Optional initial weight vector (defaults to zeros)

    Returns:
        theta: Fitted parameter vector of shape (d+1,)
        cost_history: List of J(theta) recorded after every iteration (length n_iters)
    """
    n, d_plus_1 = X.shape
    if initial_theta is not None:
        theta = np.array(initial_theta, dtype=float).copy()
    else:
        theta = np.zeros(d_plus_1, dtype=float)

    cost_history = []
    for _ in range(n_iters):
        # Compute gradient: (1/n) * X^T (h_theta(X) - y)
        errors = hypothesis(X, theta) - y
        gradient = np.dot(X.T, errors) / n
        theta = theta - alpha * gradient

        # Record J(theta) after every iteration
        cost_history.append(cost(X, y, theta))

    return theta, cost_history


def fit_sgd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_epochs: int,
    initial_theta: np.ndarray = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Fit linear regression using Stochastic Gradient Descent (row-by-row updates).

    Formula (Task 4.6):
        In each epoch loop over training rows one at a time and update:
            theta_j := theta_j + alpha * (y^(i) - h_theta(x^(i))) * x_j^(i)
        In vector notation for row i:
            theta := theta - alpha * (h_theta(x^(i)) - y^(i)) * x^(i)

    Args:
        X: Design matrix of shape (n, d+1)
        y: Target vector of shape (n,)
        alpha: Learning rate parameter
        n_epochs: Total number of passes over the training dataset
        initial_theta: Optional initial weight vector (defaults to zeros)

    Returns:
        theta: Fitted parameter vector of shape (d+1,)
        cost_history: List of J(theta) recorded after every epoch (length n_epochs)
    """
    n, d_plus_1 = X.shape
    if initial_theta is not None:
        theta = np.array(initial_theta, dtype=float).copy()
    else:
        theta = np.zeros(d_plus_1, dtype=float)

    cost_history = []
    for _ in range(n_epochs):
        # Iterate over all training rows one by one
        for i in range(n):
            xi = X[i]
            yi = y[i]
            error = np.dot(xi, theta) - yi
            theta = theta - alpha * error * xi

        # Record full dataset cost J(theta) after every epoch
        cost_history.append(cost(X, y, theta))

    return theta, cost_history


def rmse(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    clip_negative: bool = True,
) -> float:
    """
    Compute Root Mean Squared Error (RMSE) on the dataset.

    Formula (Section 3.1 & Task 4.9):
        RMSE = sqrt( 1/m * sum_{i=1}^m (y^(i) - y_pred^(i))^2 )
        Negative predictions clipped at 0 as physical solar plants cannot generate negative power.

    Args:
        X: Design matrix of shape (m, d+1)
        y: Ground truth target vector of shape (m,)
        theta: Fitted weight vector of shape (d+1,)
        clip_negative: Whether to clip negative predictions at 0 (default True)

    Returns:
        float: RMSE metric in kW
    """
    m = len(y)
    if m == 0:
        return 0.0
    preds = hypothesis(X, theta)
    if clip_negative:
        preds = np.maximum(0.0, preds)
    squared_errors = (y - preds) ** 2
    return float(np.sqrt(np.sum(squared_errors) / m))


if __name__ == "__main__":
    # Unit self-test with synthetic data
    print("Testing regression functions...")
    np.random.seed(42)
    X_test = np.c_[np.ones(100), np.random.randn(100, 3)]
    true_theta = np.array([5.0, -2.0, 3.5, 1.2])
    y_test = X_test @ true_theta + 0.1 * np.random.randn(100)

    # Test hypothesis and cost
    h = hypothesis(X_test, true_theta)
    c = cost(X_test, y_test, true_theta)
    print(f"Sample cost: {c:.6f}")

    # Test normal equation
    theta_norm = fit_normal(X_test, y_test)
    print(f"Normal eq theta: {theta_norm}")

    # Test batch GD
    theta_bgd, hist_bgd = fit_batch_gd(X_test, y_test, alpha=0.1, n_iters=500)
    print(f"Batch GD theta:  {theta_bgd}")
    print(f"Max diff:        {np.max(np.abs(theta_norm - theta_bgd)):.6f}")

    # Test SGD
    theta_sgd, hist_sgd = fit_sgd(X_test, y_test, alpha=0.01, n_epochs=50)
    print(f"SGD theta:       {theta_sgd}")

    # Test RMSE
    r = rmse(X_test, y_test, theta_norm)
    print(f"RMSE:            {r:.6f}")
    print("All unit tests passed successfully!")
