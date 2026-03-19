import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("03_Clustering_Marketing 2.csv")

# -------------------------------
# Data Cleaning
# -------------------------------

# Gender cleaning
df['gender'] = df['gender'].replace('NA', np.nan)
df['gender'] = df['gender'].fillna(df['gender'].value_counts().idxmax())
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# Age cleaning
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

# Remove outliers
df = df[df['NumberOffriends'] < 5000]

# Drop unnecessary column
if 'gradyear' in df.columns:
    df_clustering = df.drop(columns=['gradyear'])
else:
    df_clustering = df.copy()

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

# -------------------------------
# KMeans Clustering
# -------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("KMeans Silhouette Score:", silhouette_score(X_scaled, kmeans_labels))

# -------------------------------
# Hierarchical Clustering
# -------------------------------
hier = AgglomerativeClustering(n_clusters=4)
hier_labels = hier.fit_predict(X_scaled)

print("Hierarchical Silhouette Score:", silhouette_score(X_scaled, hier_labels))

# -------------------------------
# DBSCAN
# -------------------------------
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X_scaled)

print("DBSCAN Silhouette Score:", silhouette_score(X_scaled, db_labels))

# -------------------------------
# PCA Visualization
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Student Clusters Visualization")
plt.show()

# -------------------------------
# Cluster Insights
# -------------------------------
df['cluster'] = kmeans_labels

cluster_profile = df.groupby('cluster').mean()

print("\nCluster Profile:")
print(cluster_profile)

# Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(cluster_profile, cmap="coolwarm")
plt.title("Cluster Feature Profiles")
plt.show()
