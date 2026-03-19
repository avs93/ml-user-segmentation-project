#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Student Interest Segmentation Using Machine Learning

## Project Overview

Understanding user interests is fundamental for building personalized digital products. Platforms that serve students—such as learning communities, social platforms, or campus apps—need ways to group users with similar behaviors in order to deliver relevant experiences.

This project applies **machine learning clustering techniques** to segment students based on their interests and behavioral attributes extracted from social network profiles.

The objective is to identify meaningful student groups that could inform **product personalization strategies**, community recommendations, and engagement initiatives.

---

## Problem Statement

Students often express multiple interests across areas such as sports, music, fashion, and social activities. Without structured segmentation, platforms struggle to tailor experiences effectively.

The goal of this project is to:

* Segment students into meaningful groups based on their interests
* Identify behavioral patterns across student profiles
* Evaluate multiple clustering algorithms to determine the most suitable segmentation approach

---

## Dataset

The dataset contains approximately **15,000 student social network profiles** with over **40 behavioral features**, including:

* Demographic attributes (age, gender)
* Social network metrics (number of friends)
* Interest indicators such as:

  * sports activities
  * music preferences
  * fashion and shopping
  * religion
  * lifestyle behaviors

These variables represent the frequency of keywords associated with student interests.

---

## Methodology

### 1. Data Preprocessing

Before applying clustering algorithms, the dataset required several preprocessing steps:

* Handling missing values in demographic features
* Converting categorical variables (e.g., gender) into numeric form
* Converting string-based numeric fields to numeric types
* Standardizing features to ensure equal scale across variables

Feature scaling was applied using **standardization** to prevent high-magnitude variables from dominating clustering results.

---

### 2. Exploratory Data Analysis

Exploratory analysis revealed several patterns:

* Most interest variables are sparse (many zero values)
* Students often exhibit overlapping interests
* Certain categories (sports, music, shopping) appear more frequently

These observations suggest that student personas are **not mutually exclusive**, which impacts cluster separation.

---

### 3. Clustering Algorithms Evaluated

Three clustering algorithms were tested to evaluate segmentation quality:

**K-Means Clustering**
Centroid-based clustering that partitions observations into k groups.

**Hierarchical Clustering**
Builds clusters by progressively merging similar observations.

**DBSCAN**
Density-based clustering that identifies clusters based on density of points.

Cluster quality was evaluated using the **Silhouette Score**, which measures how well observations fit within their assigned cluster relative to other clusters.

---

## Results

| Algorithm    | Silhouette Score |
| ------------ | ---------------- |
| K-Means      | **0.131**        |
| Hierarchical | 0.056            |
| DBSCAN       | -0.323           |

### Observations

* **K-Means performed best**, producing moderately separated clusters.
* Hierarchical clustering showed weaker cluster separation.
* DBSCAN struggled due to the high dimensionality and sparsity of the dataset.

This outcome is consistent with behavioral datasets where user interests overlap significantly.

---

## Cluster Insights

The clustering analysis revealed several meaningful student personas:

**Sports Enthusiasts**
Students frequently mentioning basketball, football, and other sports.

**Music & Entertainment Group**
Profiles showing high engagement with music, bands, and entertainment activities.

**Fashion & Social Lifestyle Segment**
Students interested in shopping, clothing, and lifestyle topics.

**Religious Community Segment**
Profiles showing strong association with religious keywords.

**Mixed Interest Students**
Students with balanced interests across multiple categories.

---

## Product Implications

From a product perspective, these clusters could support several features:

**Personalized Content Feeds**
Recommend content aligned with dominant student interests.

**Community Discovery**
Suggest student groups or clubs based on cluster membership.

**Event Recommendations**
Promote sports events, music concerts, or campus activities tailored to interest groups.

**Targeted Engagement Campaigns**
Segment outreach campaigns based on student personas.

---

## Key Learnings

This project highlights several important considerations when applying machine learning to user segmentation:

* Behavioral data often produces **overlapping clusters**
* Feature sparsity can impact clustering performance
* Model evaluation is critical to selecting appropriate segmentation methods
* Interpretability is essential when translating ML outputs into product decisions

---

## Future Improvements

Future enhancements could include:

* Applying dimensionality reduction techniques to improve clustering
* Using feature selection to remove noisy variables
* Incorporating behavioral activity data (e.g., app usage)
* Building dynamic clustering pipelines for real-time segmentation

---

## Conclusion

This project demonstrates how machine learning clustering techniques can uncover meaningful patterns in student interest data. While the clusters are not perfectly separated, the segmentation provides useful insights that could support personalization and engagement strategies in student-focused digital products.

The exercise illustrates the practical application of machine learning to transform raw behavioral data into actionable product insights.


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.decomposition import PCA


# In[4]:


df = pd.read_csv("03_Clustering_Marketing 2.csv")

df.head()


# In[28]:


df.info()
df.describe()


# In[30]:


df.isnull().sum()


# In[32]:


df['gender'].fillna(df['gender'].mode()[0], inplace=True)

df['age'].fillna(df['age'].median(), inplace=True)


# In[34]:


df['gender'].value_counts()


# In[36]:


df['gender'].unique()


# In[38]:


df['gender'].fillna(df['gender'].mode().iloc[0], inplace=True)


# In[40]:


df['gender'].isnull().sum()


# In[42]:


df['gender'].replace('NA', np.nan, inplace=True)


# In[44]:


df['gender'] = df['gender'].replace('NA', np.nan)


# In[46]:


# convert 'NA' string to actual missing values
df['gender'] = df['gender'].replace('NA', np.nan)

# fill missing values with most common gender
df['gender'] = df['gender'].fillna(df['gender'].mode().iloc[0])

# convert gender to numeric
df['gender'] = df['gender'].map({'F':0, 'M':1})


# In[48]:


df['gender'].unique()


# In[50]:


df['gender'].value_counts(dropna=False)


# In[52]:


df['gender'].unique()


# In[54]:


# remove spaces
df['gender'] = df['gender'].str.strip()

# convert NA-like values to NaN
df['gender'] = df['gender'].replace(['NA','na','NaN','nan',''], np.nan)


# In[56]:


df['gender'] = df['gender'].astype(str)


# In[58]:


df['gender'] = df['gender'].str.strip()


# In[60]:


df['gender'] = df['gender'].replace(['NA','nan','NaN',''], np.nan)


# In[62]:


df['gender'] = df['gender'].replace(['NA','nan','NaN',''], np.nan)


# In[64]:


df['gender'] = df['gender'].map({'F':0,'M':1})


# In[66]:


df['gender'].unique()


# In[68]:


df['gender'].value_counts(dropna=False)


# In[70]:


df['gender'].unique()


# In[72]:


import numpy as np

# convert to string
df['gender'] = df['gender'].astype(str)

# normalize values
df['gender'] = df['gender'].str.strip().str.upper()

# replace NA-like values
df['gender'] = df['gender'].replace(['NA','NAN','NONE',''], np.nan)

# fill missing values with most common gender
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])


# In[74]:


df['gender'].head(20)


# In[76]:


df['gender'].value_counts(dropna=False)


# In[78]:


df = pd.read_csv("03_Clustering_Marketing 2.csv")


# In[80]:


import numpy as np

df['gender'] = df['gender'].replace('NA', np.nan)


# In[82]:


df['gender'] = df['gender'].fillna(df['gender'].value_counts().idxmax())


# In[84]:


df['gender'] = df['gender'].replace({'F':0,'M':1})


# In[86]:


df['gender'].unique()


# In[88]:


df['gender'].isnull().sum()


# In[92]:


df['gender'].unique()


# In[94]:


df['gender'].dtype


# In[96]:


df['age'].isnull().sum()


# In[98]:


df['age'] = df['age'].fillna(df['age'].median())


# In[100]:


df['age'] = pd.to_numeric(df['age'], errors='coerce')


# In[102]:


df['age'] = df['age'].fillna(df['age'].median())


# In[104]:


df['age'].dtype


# In[106]:


df['age'].isnull().sum()


# In[108]:


# convert age to numeric
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# fill missing values with median
df['age'] = df['age'].fillna(df['age'].median())


# In[110]:


df.describe()


# In[112]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.heatmap(df.corr())
plt.show()


# In[116]:


df.hist(figsize=(15,10))
plt.show()


# In[118]:


df.skew()


# In[120]:


plt.figure(figsize=(10,6))
sns.boxplot(data=df[['age','NumberOffriends']])
plt.show()


# In[122]:


df = df[df['NumberOffriends'] < 5000]


# In[124]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)


# In[126]:


from sklearn.cluster import KMeans

wcss = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[128]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)

kmeans_labels = kmeans.fit_predict(X_scaled)

df['kmeans_cluster'] = kmeans_labels


# In[130]:


from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, kmeans_labels)

print("Silhouette Score:", score)


# In[132]:


df_clustering = df.drop(columns=['gradyear'])


# In[134]:


df_clustering.sum().sort_values()


# In[136]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df_clustering)


# In[138]:


kmeans = KMeans(n_clusters=4, random_state=42)

labels = kmeans.fit_predict(X_scaled)


# In[140]:


from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, labels)

print(score)


# In[142]:


silhouette_score(X_scaled, kmeans_labels)


# In[144]:


from sklearn.cluster import AgglomerativeClustering

hier = AgglomerativeClustering(n_clusters=4)

hier_labels = hier.fit_predict(X_scaled)

from sklearn.metrics import silhouette_score

silhouette_score(X_scaled, hier_labels)


# In[146]:


from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)

db_labels = db.fit_predict(X_scaled)

silhouette_score(X_scaled, db_labels)


# In[151]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='viridis')

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Student Clusters Visualization")

plt.show()


# In[153]:


cluster_profile = df.groupby('kmeans_cluster').mean()

cluster_profile


# In[155]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

sns.heatmap(cluster_profile, cmap="coolwarm")

plt.title("Cluster Feature Profiles")

plt.show()


# In[ ]:





# In[ ]:




