import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

st.set_page_config(layout="wide")

# -------- COMPACT STYLE FOR SCREENSHOT --------
st.markdown("""
<style>
.block-container {padding-top: 0.5rem; padding-bottom: 0rem;}
h1 {font-size:26px;}
h2 {font-size:20px;}
h3 {font-size:16px;}
p {font-size:13px;}
</style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.title("🟣 News Topic Discovery Dashboard")
st.caption("Hierarchical Clustering for automatic news grouping")

# -------- SIDEBAR --------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")
else:
    st.stop()

text_col = st.sidebar.selectbox("Select text column", df.columns)
text_data = df[text_col].dropna()

# -------- TFIDF --------
st.sidebar.header("Text Vectorization")

max_features = st.sidebar.slider("Max TF-IDF Features", 100, 2000, 1000)
use_stop = st.sidebar.checkbox("Remove Stopwords", True)

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stop else None
)

X = vectorizer.fit_transform(text_data).toarray()

# -------- CLUSTER CONTROLS --------
st.sidebar.header("Clustering")
linkage_method = st.sidebar.selectbox("Linkage", ["ward","complete","average","single"])
num_clusters = st.sidebar.slider("Number of clusters", 2, 10, 5)

# -------- DENDROGRAM --------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dendrogram")
    fig1, ax1 = plt.subplots(figsize=(5,3))
    sch.dendrogram(sch.linkage(X[:120], method=linkage_method))
    plt.title("Hierarchical Tree")
    st.pyplot(fig1)

# -------- CLUSTER MODEL --------
hc = AgglomerativeClustering(
    n_clusters=num_clusters,
    metric="euclidean",
    linkage=linkage_method
)
labels = hc.fit_predict(X)

# -------- PCA REDUCTION --------
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# -------- CLUSTER GRAPH --------
with col2:
    st.subheader("Cluster Graph")
    fig2, ax2 = plt.subplots(figsize=(5,3))

    colors = ['red','blue','green','purple','orange','cyan','yellow','pink']

    for i in range(num_clusters):
        ax2.scatter(
            X_2d[labels==i,0],
            X_2d[labels==i,1],
            s=20,
            c=colors[i],
            label=f"Cluster {i}"
        )

    ax2.legend(fontsize=6)
    st.pyplot(fig2)

# -------- SILHOUETTE --------
score = silhouette_score(X, labels)
st.metric("Silhouette Score", round(score,3))

# -------- SUMMARY --------
st.subheader("Topic Summary")
terms = vectorizer.get_feature_names_out()

for c in range(num_clusters):
    cluster_texts = text_data[labels==c]
    if len(cluster_texts)==0:
        continue

    tfidf_mean = X[labels==c].mean(axis=0)
    top_idx = np.argsort(tfidf_mean)[-5:]
    words = [terms[i] for i in top_idx]

    st.write(f"Cluster {c}: ", ", ".join(words))

# -------- INSIGHT --------
st.info(
"Articles grouped in the same cluster share similar vocabulary and themes. "
"Useful for automatic tagging, recommendations, and topic discovery."
)