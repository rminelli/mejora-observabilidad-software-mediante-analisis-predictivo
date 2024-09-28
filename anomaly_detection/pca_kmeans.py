# pca_kmeans.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate simulated data (replace this with your actual data)
np.random.seed(42)
data = np.random.normal(size=(100, 5))
anomalies = np.random.uniform(low=-5, high=5, size=(5, 5))
data_with_anomalies = np.vstack([data, anomalies])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_with_anomalies)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)

# Apply K-Means
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(pca_data)

# Visualization
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.title("Anomaly Detection using PCA and K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
