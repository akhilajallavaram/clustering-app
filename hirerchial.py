import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/NelakurthiSudheer/Mall-Customers-Segmentation/main/Dataset/Mall_Customers.csv')
data = df[["Annual Income (k$)","Spending Score (1-100)"]]
def plot_dendogram(X, method='ward'):

    scalar = StandardScaler()
    X = scalar.fit_transform(data)
    clustering = AgglomerativeClustering(n_clusters=5, linkage=method)
    clustering.fit(X)
    score = silhouette_score(X, clustering.labels_)
    st.write(f"Silhouette score for {method} linkage: {score:.3f}")

    """Plot the dendrogram for the given data using the specified clustering method."""
    Z = linkage(X, method=method)
    # Plot the dendrogram
    fig = plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f'Dendrogram ({method} linkage)')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(fig)

def hirerechial():
    st.title('Agglomerative Clustering with Dendrogram')
    method = st.selectbox('Select the clustering method', ('ward', 'complete', 'average', 'single'))
    plot_dendogram(data, method)


if __name__ == '__main__':
    hirerechial()
