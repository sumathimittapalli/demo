# -----------------------------
# K-Means Clustering Example
# -----------------------------

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -----------------------------
# Sample Data
# -----------------------------
x = np.array([
    [1,2], [1,3], [1,4], [1,0],
    [10,2], [10,1], [10,3], [10,4],
    [5,2], [5,1], [5,3], [5,0],
    [8,2], [8,1], [8,3], [8,0]
])

# -----------------------------
# Apply K-Means
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(x)

# Print results
print("Cluster Labels:", labels)
print("Centroids:\n", kmeans.cluster_centers_)

# -----------------------------
# Visualization of Clusters
# -----------------------------
plt.scatter(x[:, 0], x[:, 1], c=labels, s=100)
plt.scatter(
    kmeans.cluster_centers_[:, 0], 
    kmeans.cluster_centers_[:, 1],
    s=300, marker='X', c='red'  # Centroids
)
plt.title("K-Means Clustering")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
