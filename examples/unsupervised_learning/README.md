# Unsupervised Learning

## Overview

**Unsupervised learning** studies data when there are **no response labels provided**. Rather than learning from known target values, these methods look for organization, recurring patterns, hidden relationships, and overall structure already present in the observations.

This folder includes a set of **from-scratch implementations and example notebooks** covering major unsupervised learning approaches. Each notebook explores a different way of describing structure in data, such as clustering by distance, grouping by density, reducing dimensionality through variance, or detecting communities through graph links.

The notebooks are written to highlight **intuition**, **mathematical reasoning**, and **clear interpretation**, instead of depending entirely on black-box tools.

---

## Contents

The `unsupervised_learning` directory contains the following notebooks, each organized in its own subfolder:

### K-Means Clustering
A centroid-based clustering method that separates data into a chosen number of groups by reducing within-cluster spread.

**Key ideas:**
- Distance-based clustering
- Requires specifying the number of clusters
- Sensitive to feature scaling
- Best suited for roughly spherical clusters

---

### DBSCAN Clustering
A density-based clustering method that forms groups from dense local regions and marks isolated observations as noise.

**Key ideas:**
- Density-based clustering
- No need to specify the number of clusters
- Detects outliers naturally
- Handles non-spherical cluster shapes

---

### Principal Component Analysis (PCA)
An unsupervised dimension reduction method that maps data onto orthogonal directions that capture the greatest variability.

**Key ideas:**
- Variance-based dimensionality reduction
- Identifies latent structure in high-dimensional data
- Useful for visualization, noise reduction, and preprocessing
- Preserves variance, not class separation

---

### Community Detection (Label Propagation)
A network-based method that finds communities from connection patterns instead of ordinary feature similarity.

**Key ideas:**
- Operates on graph structure, not feature space
- Discovers groups through local label consensus
- No predefined number of communities
- Well-suited for network and relational data

---

## Learning Themes

Across all notebooks, the main ideas include:

- Finding meaningful structure without labeled outputs  
- Understanding how geometry, density, variance, and connectivity shape the analysis  
- Recognizing the importance of preprocessing and representation of the data  
- Comparing the advantages and weaknesses of different unsupervised techniques  
- Focusing on interpretation of patterns, not only algorithmic performance  

Together, these notebooks show that each method uses its own notion of similarity, so the best choice depends on the type of structure present in the data.

---

## Folder Organization

unsupervised_learning/
├── Community Detection/
│   └── community_detection_example.ipynb
├── DBSCAN/
│   └── dbscan_example.ipynb
├── K-Means Clustering/
│   └── k_means_clustering.ipynb
├── PCA/
│   └── pca_example.ipynb