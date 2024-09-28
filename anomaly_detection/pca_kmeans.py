import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_with_anomalies = pd.read_csv('../data/anomaly_detection_data.csv')

# Create a pandas DataFrame
columns = [f'feature_{i}' for i in range(1, 6)]  # Create feature names
df = pd.DataFrame(data_with_anomalies, columns=columns)

# Display the first few rows of the DataFrame
print("Data with Anomalies (First 5 Rows):")
print(df.head())

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)

# Apply K-Means with explicit n_init parameter
kmeans = KMeans(n_clusters=2, n_init=10)  # Explicitly set n_init to 10
clusters = kmeans.fit_predict(pca_data)

# Add PCA and Cluster information to the DataFrame
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]
df['Cluster'] = clusters

# Display the updated DataFrame with PCA components and cluster assignments
print("\nData with PCA and Cluster Assignments (First 5 Rows):")
print(df.head())

# Visualization
plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')
plt.title("Anomaly Detection using PCA and K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
