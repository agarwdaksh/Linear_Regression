# Simple Linear Regression Implementation

## Overview
This repository contains Python implementation of Simle Linear Regression model using the gradient desceent algorithm. Here, I have focused on implementing the model from scratch without using libraries like scikit-learn.

## Repository Contents

*   `linearX.csv`:  The independent/predictor variable dataset.
*   `linearY.csv`: The dependent/response variable dataset.
*   `Linear_Regression.ipynb`: A Jupyter Notebook containing the Python code for the linear regression model and experiments.
*   `README.md`: This file, providing an overview of the project.

## Libraries Used

*   `numpy`: For numerical computations and array manipulations.
*   `pandas`: For reading and manipulating the dataset.
*   `matplotlib`: For plotting graphs and visualizations.

## Code Explanation

The core of the implementation resides within the `Linear_Regression.ipynb` notebook. Here's a breakdown:

1.  **Data Loading and Preprocessing:**
    *   The notebook uses `numpy` and `pandas` to load the data from the provided CSV files (`linearX.csv` and `linearY.csv`).
    *   Data normalization (mean and standard deviation scaling) is applied to the predictor variable to improve the convergence of the gradient descent algorithm.
    *   A column of ones is added to the predictor variable (`x`) to account for the intercept term (theta_0).
2.  **Cost Function:**
    *   The cost function used is the Mean Squared Error (MSE),  averaging the cost as `J(θ₀, θ₁) = (1 / 2m) * Σᵢ₌₁ᵐ (hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²`.
3.  **Gradient Descent:**
    *   Batch gradient descent is implemented to update the parameters (theta\_0 and theta\_1) iteratively.
    *   The code includes a convergence check based on the change in the cost function between iterations.
4.  **Stochastic and Mini-Batch Gradient Descent:**
    *   Functions for stochastic gradient descent and mini-batch gradient descent are implemented.
5.  **Experimentation and Results:**
    *   The notebook explores the impact of different learning rates on the cost function's convergence.
    *   Plots are generated to visualize the cost function's behavior over iterations for different learning rates and gradient descent methods.
    *   A scatter plot of the dataset and the resulting regression line are plotted.

## Results and Observations

*   **Convergence:** The batch gradient descent algorithm converges to a minimum cost function value. The convergence rate is highly dependent on the learning rate.
*   **Learning Rate:**  A suitable learning rate needs to be chosen to ensure convergence.  Too high and the algorithm will diverge; too low and the algorithm will take a long time to converge.
*   **Gradient Descent Methods:** Stochastic and mini-batch gradient descent exhibit different convergence behaviors compared to batch gradient descent.  They typically have more noisy convergence paths but can converge faster in some cases.

