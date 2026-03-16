# Student Interest Segmentation Using Machine Learning

## Project Overview

Understanding user interests is critical for building personalized digital experiences. Platforms serving student communities often struggle to tailor content, events, and recommendations due to limited insight into user behavior.

This project applies **machine learning clustering techniques** to segment students based on their interests and behavioral attributes extracted from social network profiles.

The objective is to discover meaningful student groups that could support **personalized product experiences**, targeted engagement strategies, and community recommendations.

---

## Problem Statement

Students often express multiple interests such as sports, music, fashion, and social activities. Without structured segmentation, platforms cannot effectively personalize experiences.

This project explores:

* Can machine learning automatically identify groups of students with similar interests?
* Which clustering algorithm produces the most meaningful segmentation?

---

## Dataset

The dataset contains approximately **15,000 student profiles** with over **40 behavioral attributes**, including:

* Demographics

  * Age
  * Gender
* Social activity

  * Number of friends
* Interest indicators such as:

  * Sports
  * Music
  * Shopping
  * Religion
  * Lifestyle activities

Each feature represents the frequency of interest-related keywords appearing in student profiles.

---

## Project Workflow

### 1. Data Preprocessing

The dataset required several preprocessing steps before applying machine learning models:

* Handling missing values in demographic attributes
* Converting categorical variables (gender) into numeric format
* Converting string-based numeric fields (age) into numeric data types
* Standardizing features to ensure equal contribution across variables

Feature scaling was performed using **Standardization** to improve clustering performance.

---

### 2. Exploratory Data Analysis

EDA helped uncover patterns in the dataset:

* Many features are sparse (mostly zero values)
* Student interests often overlap across multiple categories
* Some interest categories appear significantly more frequently than others

These insights helped guide feature preparation for clustering.

---

### 3. Clustering Algorithms Evaluated

Three clustering approaches were tested:

**K-Means Clustering**
A centroid-based algorithm that partitions observations into k clusters.

**Hierarchical Clustering**
Builds clusters by progressively merging similar observations.

**DBSCAN**
A density-based algorithm that groups points based on density patterns.

---

## Model Evaluation

Cluster quality was evaluated using **Silhouette Score**, which measures how well data points fit within their assigned clusters compared to other clusters.

| Algorithm    | Silhouette Score |
| ------------ | ---------------- |
| K-Means      | **0.131**        |
| Hierarchical | 0.056            |
| DBSCAN       | -0.323           |

### Key Observation

K-Means clustering produced the best segmentation among the tested algorithms. While the score indicates moderate overlap between clusters, this is expected due to the natural overlap in student interests.

---

## Cluster Insights

The clustering analysis revealed several meaningful student personas:

**Sports Enthusiasts**
Students frequently mentioning basketball, football, and soccer.

**Music & Entertainment Group**
Profiles showing strong engagement with music, bands, and entertainment activities.

**Fashion & Lifestyle Segment**
Students interested in shopping, clothing, and social lifestyle topics.

**Religious Community Segment**
Profiles frequently referencing church and spiritual topics.

**Mixed Interest Students**
Students with diverse interests across multiple categories.

---

## Visualization

Clusters were visualized using **Principal Component Analysis (PCA)** to reduce the dataset to two dimensions and visually represent cluster separation.

This helps interpret how student segments emerge from the behavioral data.

---

## Product Applications

From a product perspective, this segmentation could support several features:

**Personalized Content Feeds**
Recommend relevant content based on dominant student interests.

**Community Discovery**
Suggest student groups aligned with user interests.

**Event Recommendations**
Promote sports events, concerts, or campus activities tailored to each segment.

**Targeted Engagement Campaigns**
Segment outreach strategies based on user personas.

---

## Key Learnings

This project highlights several important aspects of applying machine learning to behavioral data:

* User interests often overlap, leading to imperfect cluster separation
* Feature sparsity can impact clustering quality
* Model evaluation is essential when selecting clustering techniques
* Interpreting clusters is critical to translating ML outputs into product insights

---

## Future Improvements

Potential enhancements include:

* Applying dimensionality reduction techniques for improved clustering
* Feature selection to remove noisy variables
* Incorporating additional behavioral signals such as platform activity
* Developing dynamic clustering pipelines for real-time segmentation

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Author

Senior Product Manager transitioning into **AI Product Management**, exploring how machine learning can power data-driven product experiences.

This repository documents hands-on experimentation with ML techniques and their application to real-world product problems.
