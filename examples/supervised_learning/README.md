# Supervised Learning Notebooks

This folder contains Jupyter notebooks that showcase **supervised learning
algorithms built from scratch** for the INDE577 project.

Each notebook walks through a complete machine learning process, including:

- Loading and preparing data
- Performing exploratory data analysis (EDA)
- Training the model
- Plotting and examining model behavior
- Assessing results and discussing interpretation

All algorithms in these notebooks come from the project’s custom machine
learning package. Standard scikit-learn classifiers and regressors are not used.

---

## What Is Supervised Learning?

Supervised learning is a branch of machine learning in which models are trained
on **labeled examples**, where each data point has input variables together
with a known output.

The objective is to learn a relationship between features and targets so that
the model can make useful predictions on new data.

Supervised learning is usually divided into two broad types:

- **Classification**: predicting category labels
- **Regression**: predicting numeric outcomes

The notebooks in this folder demonstrate how these methods can be constructed
from basic principles and applied to real datasets.

---

## Included Notebooks and Methods

Below is a summary of the supervised learning methods included in this folder,
grouped by subdirectory.

---

### Decision Tree

**Folder:** `Decision Tree/`  
**Notebook:** `decision_tree_ex.ipynb`

This notebook covers:

- Constructing a decision tree classifier from scratch
- Using impurity measures to choose splits
- Understanding recursive partitioning
- Examining overfitting and tree depth limits
- Measuring classification performance

---

### Ensemble Methods

**Folder:** `Ensemble Methods/`  
**Notebook:** `ensemble_methods_example.ipynb`

This notebook studies ensemble techniques formed by combining simpler models.

Topics include:

- Bagging and random forests
- Reducing variance through aggregation
- Comparing ensemble models with a single tree
- Studying predictive stability and accuracy

The notebook also shows why combining multiple learners can outperform one
individual model alone.

---

### Gradient Descent

**Folder:** `Gradient Descent/`  
**Notebook:** `gradient_descent_example.ipynb`

This notebook emphasizes optimization rather than one particular algorithm.

Topics include:

- Deriving gradient descent from first principles
- Learning rate choices and convergence
- Visualizing how loss decreases
- Linking gradient descent to linear and logistic regression

---

### k-Nearest Neighbors (KNN)

**Folder:** `knn_example/`  
**Notebook:** `knn_example.ipynb`  
**Dataset:** `iris.csv`

This notebook demonstrates:

- Classification based on nearest neighbors
- The role of scaling and distance measures
- How the choice of \( k \) changes results
- Visualizing decision regions
- Accuracy on the Iris dataset

---

### Linear Regression

**Folder:** `Linear Regression/`  
**Notebook:** `linear_regression_boston.ipynb`

This notebook demonstrates:

- Linear regression trained with gradient descent
- Predicting continuous responses
- Interpreting fitted coefficients
- Plotting regression behavior
- Using error-based evaluation measures

---

### Logistic Regression

**Folder:** `Logistic Regression/`  
**Notebook:** `logistic_regression_pima.ipynb`

This notebook includes:

- Binary classification with logistic regression
- From-scratch training with gradient descent
- Understanding probabilities and coefficients
- Evaluating results with accuracy and similar metrics

---

### Multilayer Perceptron

**Folder:** `Multilayer Perceptron/`  
**Notebook:** `multilayer_perceptron_example.ipynb`

This notebook presents:

- A feedforward neural network written from scratch
- Forward pass and backpropagation
- Nonlinear activation functions
- Learning behavior during training
- Comparison of training and testing performance

---

### Perceptron

**Folder:** `Perceptron/`  
**Notebook:** `perceptron_example.ipynb`

This notebook demonstrates:

- Binary classification with a single-layer perceptron
- The perceptron learning rule
- Convergence behavior during training
- The limits of linear separating boundaries

---

### Regression Trees

**Folder:** `Regression Trees/`  
**Notebook:** `regression_trees_example.ipynb`

This notebook uses a tree-based model on a dataset with a continuous response.

Topics include:

- Applying tree methods to continuous targets
- CART-style recursive splitting
- Measuring prediction quality
- Investigating how tree depth affects performance

---

## Purpose of This Directory

These notebooks are designed to:

- Show how to use the project’s supervised learning models correctly
- Provide reproducible examples for INDE577 coursework
- Build intuition for core machine learning ideas
- Act as references for checking implementation behavior

Taken together, the notebooks support the main source code and testing files by
showing how each method performs in practice.

---

## Notes

- Every supervised model in this project is implemented from scratch.
- Scikit-learn supervised algorithms are not used.
- Visualizations are produced with Matplotlib.
- Datasets are either saved in the relevant folders or loaded from public
  sources.
- Mathematical notation is written in GitHub-friendly LaTeX.

## Folder Organization

supervised_learning/
├── Decision Tree/
│   └── decision_tree_ex.ipynb
├── Ensemble Methods/
│   └── ensemble_methods_example.ipynb
├── Gradient Descent/
│   └── gradient_descent_example.ipynb
├── KNN/
│   └── knn_example.ipynb
├── Linear Regression/
│   └── linear_regression_boston.ipynb
├── Logistic Regression/
│   └── logistic_regression_pima.ipynb
├── Multilayer Perceptron/
│   └── multilayer_perceptron_example.ipynb
├── Perceptron/
│   └── perceptron_example.ipynb
├── Regression Trees/
│   └── regression_trees_example.ipynb