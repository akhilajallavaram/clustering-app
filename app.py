import streamlit as st
from clustering import k_means
from hirerchial import hirerechial


page = st.sidebar.selectbox("select type of clustering ", ("k-means", "hirerchial"))

if page == "k-means":
    k_means()
else:
    hirerechial()