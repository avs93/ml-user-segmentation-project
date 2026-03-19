import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Title
st.title("🎯 Student Interest Segmentation")

st.write("This app segments students based on their interests using Machine Learning.")

# Load dataset
df = pd.read_csv("03_Clustering_Marketing 2.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Data Cleaning
# -------------------------------
df['gender'] = df['gender'].replace('NA', np.nan)
df['gender'] = df['gender'].fillna(df['gender'].value_counts().idxmax())
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

# Drop non-useful column
if 'gradyear' in df.columns:
    df = df.drop(columns=['gradyear'])

# -------------------------------
# Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -------------------------------
# KMeans Clustering
# -------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# -------------------------------
# Show Cluster Distribution
# -------------------------------
st.subheader("📌 Cluster Distribution")
st.write(df['cluster'].value_counts())

# -------------------------------
# PCA Visualization
# -------------------------------
st.subheader("📉 Cluster Visualization (PCA)")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'])
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")

st.pyplot(fig)

# -------------------------------
# Cluster Insights
# -------------------------------
st.subheader("🧠 Cluster Insights")

cluster_profile = df.groupby('cluster').mean()
st.write(cluster_profile)
