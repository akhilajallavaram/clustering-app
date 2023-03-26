import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv("https://raw.githubusercontent.com/NelakurthiSudheer/Mall-Customers-Segmentation/main/Dataset/Mall_Customers.csv")
data = df[["Annual Income (k$)","Spending Score (1-100)"]]

def kmeans_cluster(x1, x2):
    """Predict the cluster for a given data point."""
    model = KMeans(n_clusters=5, random_state=0)
    model.fit(data)
    X = np.array([[x1, x2]])
    cluster = model.predict(X)[0]
    return cluster


def k_means():
    st.title('Clustering with annual income and spending score')
    # Get user input
    x1 = st.number_input('Enter Annual Income (k$)')
    x2 = st.number_input('Enter Spending Score (1-100)')
    if st.button('cluster'):
        cluster_result = kmeans_cluster(x1, x2)
        st.write(f'The data point ({x1}, {x2}) belongs to cluster {cluster_result}')




if __name__ == '__main__':
    k_means()



