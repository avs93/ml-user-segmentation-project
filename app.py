import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Student Interest Segmentation")

# load data
df = pd.read_csv("03_Clustering_Marketing 2.csv")

st.write("Dataset Preview")
st.dataframe(df.head())

# preprocessing
df['gender'] = df['gender'].replace('NA', None)
df['gender'] = df['gender'].fillna(df['gender'].value_counts().idxmax())
df['gender'] = df['gender'].map({'F':0, 'M':1})

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'] = df['age'].fillna(df['age'].median())

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# model
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# show clusters
st.write("Cluster Distribution")
st.write(df['cluster'].value_counts())
