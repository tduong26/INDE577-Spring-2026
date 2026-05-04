# Linear Regression

This directory contains a Jupyter notebook demonstrating **linear regression
implemented entirely from scratch** using the custom `rice_ml` package.

The notebook emphasizes both the **mathematical intuition** behind linear
regression and its **practical application** to a real-world dataset.

---

## 📘 Notebook Overview

**Notebook:** `linear_regression_example.ipynb`  
**Model:** `LinearRegression` (custom implementation)  
**Dataset:** Red Wine Quality Dataset (UCI Machine Learning Repository)

This notebook walks through a complete supervised learning workflow, including:

- Dataset loading and inspection
- Exploratory data analysis (EDA)
- Feature preprocessing and standardization
- Training a linear regression model using the closed-form normal equation
- Model evaluation using multiple regression metrics
- Visualization of predictions, residuals, and coefficients
- Interpretation of learned coefficients on standardized features

All models and utilities are implemented from scratch without using
scikit-learn.

---

## 📊 Dataset Description

The Red Wine Quality dataset contains **1,599 observations** describing
physicochemical properties of red *Vinho Verde* wine samples from Portugal.
Each observation consists of **11 numerical features** describing the
chemical composition of the wine.

The target variable is:

- **quality** — an expert sensory rating on an integer scale from 0 to 10

All features are continuous numerical variables, and no missing values are
present in the dataset. Feature scales vary significantly (for example,
`total sulfur dioxide` ranges in the hundreds while `density` is near 1.0),
motivating standardization prior to model training.

The quality target is technically **ordinal**, but is treated as a
continuous regression target for the purposes of this notebook.

---

## 🧠 What Is Linear Regression?

Linear regression models the relationship between a set of input features
and a continuous target variable as a linear combination of those features.

In matrix form, the model assumes:

y = Xβ + ε

where:
- `β` represents the model coefficients
- `ε` represents random noise

In this notebook, model parameters are estimated using **Ordinary Least
Squares (OLS)** via the closed-form normal equation, which minimizes the
mean squared error between predictions and observed values.

---

## 🔍 Exploratory Data Analysis

Exploratory analysis is used to:

- Examine feature distributions and summary statistics
- Understand the distribution of the quality target
- Identify linear relationships between features and the target
- Visualize correlation structure using a heatmap
- Highlight the features most strongly correlated with wine quality

This analysis provides intuition for which physicochemical properties are
likely to have the strongest influence on perceived wine quality, with
`alcohol` and `volatile acidity` emerging as particularly informative.

---

## ⚙️ Preprocessing

Before training the model, the data is:

- Separated into feature and target matrices
- Split into training and test sets using a custom implementation
- Standardized to zero mean and unit variance

Standardization improves numerical stability and allows model coefficients
to be interpreted on a common scale, since the original features differ in
scale by several orders of magnitude.

---

## 📈 Model Training and Evaluation

The linear regression model is trained using the **closed-form normal
equation**, without regularization.

Model performance is evaluated using:

- R² score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

Evaluation is performed on both training and test sets to assess
generalization.

---

## 📐 Visualization and Diagnostics

The notebook includes standard diagnostic visualizations, such as:

- Distribution of the quality target
- Feature correlation heatmap
- Feature–target correlation bar chart
- Predicted vs. actual target values
- Standardized coefficient magnitudes
- Residuals vs. predicted values
- Residual distribution histogram

These plots help assess model fit, bias, and variance, and reveal limitations
of the linear modeling assumption — including a characteristic banding
pattern in the residuals that arises from the discrete integer nature of
the quality target.

---

## ⚠️ Model Limitations

Linear regression assumes a linear relationship between features and the
target variable. While the model captures the general trend in wine quality,
the dataset contains nonlinear relationships and feature interactions
(for example, between alcohol and acidity) that a linear model cannot fully
represent.

Additionally, the quality score is an **ordinal integer** between 3 and 8,
not a true continuous variable. This causes predictions to be smoothed
toward the mean and produces residuals that cluster in integer steps.
Ordinal classification models, regression trees, or ensemble methods may
capture this structure more faithfully, but at the cost of interpretability.

---

## 🎯 Purpose of This Notebook

This notebook is designed to:

- Demonstrate linear regression implemented from first principles
- Reinforce the intuition behind regression modeling
- Provide a clear example of regression diagnostics and evaluation
- Illustrate the limitations of linear models on ordinal targets
- Serve as a reusable template for regression analysis in this project

---

## 📝 Notes

- The linear regression model is implemented from scratch in the `rice_ml`
  package.
- No scikit-learn regression models are used.
- Visualizations are created using Matplotlib.
- All mathematical expressions are written in GitHub-compatible format.

This notebook complements the project's source code and unit tests by
providing a transparent and interpretable example of linear regression in
practice.