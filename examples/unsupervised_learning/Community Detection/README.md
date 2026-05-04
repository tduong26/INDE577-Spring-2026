# Community Detection with Label Propagation

## Overview

Community detection is an **unsupervised learning** task on graph data that identifies groups of nodes which are more strongly connected to each other than to the rest of the network.

Unlike clustering in feature space, community detection works directly with **graph structure**, using edges between nodes instead of distances between feature vectors.

This notebook demonstrates **Label Propagation** from scratch using the `rice_ml` package and applies it to a real-world social network.

---

## Key Idea

Label Propagation:

- Treats each node's identity as a **label** that can spread to neighbors
- Lets every node take on the **most common label** among its neighbors
- Repeats this process until labels stop changing
- Discovers the number of communities **automatically** from the graph

---

## Mathematical Intuition

A graph consists of nodes $V$ and edges $E$. The goal is to partition the graph so that connections are dense **within** communities and sparse **between** them.

### Label Propagation Algorithm

1. **Initialize** every node with its own unique label  
2. **Update** each node's label to the most common label among its neighbors  
3. **Repeat** until labels stop changing or `max_iter` is reached  

For node $i$, the update rule is:

$$
\ell_i^{(t+1)} = \arg\max_{\ell} \sum_{j \in N(i)} \mathbf{1}(\ell_j^{(t)} = \ell)
$$

where $N(i)$ is the neighborhood of node $i$. Labels spread through dense regions, and communities emerge from the topology itself.

---

## Dataset

The notebook uses the **Facebook Large Page-Page Network** dataset from the UCI Machine Learning Repository, downloaded directly from the web in zipped form.

Two files are used:

- `musae_facebook_edges.csv` — undirected edges between pages (`id_1`, `id_2`)  
- `musae_facebook_target.csv` — page metadata including `page_type` (used only for interpretation, not for clustering)  

---

## Exploratory Analysis

For graph data, EDA focuses on **network structure** rather than feature distributions.

The notebook examines:

- Total number of nodes and edges  
- Degree statistics (min, max, mean)  
- Degree distribution (histogram)  

Key observation: most nodes have modest degree while a small number of hubs have many connections — a heavy-tailed pattern typical of real social networks. These hubs often anchor communities.

---

## Building a 500-Node Subgraph

To keep the example computationally manageable, the notebook builds an induced subgraph from the **500 highest-degree nodes**:

1. Compute degree of each node in the full graph  
2. Select the top 500 by degree  
3. Keep only edges with both endpoints in that set  
4. Remap node IDs to consecutive integers `0, ..., 499`  
5. Build a dense adjacency matrix `A` of shape `(500, 500)` with zero diagonal  

This adjacency matrix is the input to Label Propagation.

---

## Running Label Propagation

The algorithm is fit on the subgraph with `max_iter = 100`. No target number of communities is specified — the count and sizes are determined entirely by the graph's connectivity.

The output is a label vector of length 500 assigning each node to a community.

---

## Interpreting Results

Community labels are **identifiers only** — only equality matters, not numeric order. The notebook reports:

- **Community sizes** — counts of nodes per label, showing dominant vs. localized groups  
- **Degree vs. community assignment** — a scatter plot revealing that high-degree nodes often anchor communities while lower-degree nodes attach to nearby groups  

---

## Spectral Embedding for Visualization

Because the data is a graph rather than feature vectors, the notebook uses a **spectral embedding** based on the graph Laplacian.

If $A$ is the adjacency matrix and $D$ is the diagonal degree matrix, the unnormalized Laplacian is:

$$
L = D - A
$$

The eigenvectors of $L$ associated with the smallest nonzero eigenvalues provide a 2D representation that preserves graph structure. Nodes appearing close together in this embedding tend to have similar connectivity. Coloring by community label shows whether Label Propagation aligns with the structural pattern visible in the Laplacian.

---

## Comparing Communities to Page Types

Although Label Propagation does not use page metadata, the notebook compares detected communities to the known `page_type` labels via a cross-tabulation. A strong concentration of one page type inside a community suggests that the graph topology captures meaningful real-world groupings.

---

## Strengths and Limitations

### Strengths

- Does not require the number of communities to be specified  
- Computationally simple and easy to implement  
- Operates directly on graph structure with no feature engineering  

### Limitations

- Results can vary across runs due to label-update ordering  
- May produce unstable communities when the graph has weak structure  
- Does not provide a quality score (such as modularity) for community choice  
- Sensitive to dense subgraph patterns  

---

## Conclusion

This notebook demonstrates Label Propagation as a foundational unsupervised method for graph data.

Key takeaways:

- Community detection operates on graph structure instead of feature vectors  
- Label Propagation discovers communities automatically without specifying a count  
- Subgraph extraction is a practical strategy when the full graph is too large  
- Spectral embedding provides an effective way to visualize graph clusters  

Community detection complements the K-Means, DBSCAN, and PCA notebooks as the graph-data example in the unsupervised learning track of this project.