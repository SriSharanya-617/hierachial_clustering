import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.write("Automatically group similar news articles using Hierarchical Clustering")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")
else:
    st.warning("Upload dataset to continue")
    st.stop()

# Detect text column
text_col = st.sidebar.selectbox("Select text column", df.columns)

# ---------------- TF-IDF CONTROLS ----------------
st.sidebar.header("Text Vectorization")

max_features = st.sidebar.slider("Max TF-IDF features", 100, 2000, 1000)

use_stop = st.sidebar.checkbox("Remove English stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1,1)
elif ngram_option == "Bigrams":
    ngram_range = (2,2)
else:
    ngram_range = (1,2)

# ---------------- CLUSTER CONTROLS ----------------
st.sidebar.header("Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage method",
    ["ward", "complete", "average", "single"]
)

distance_metric = "euclidean"

sample_size = st.sidebar.slider("Articles for dendrogram", 20, 200, 100)

# ---------------- TF-IDF ----------------
text_data = df[text_col].dropna()

vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stop else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(text_data).toarray()

# ---------------- DENDROGRAM ----------------
st.header("🌳 Dendrogram")

if st.button("Generate Dendrogram"):

    sample = X[:sample_size]

    fig, ax = plt.subplots(figsize=(10,5))
    sch.dendrogram(sch.linkage(sample, method=linkage_method))
    plt.title("Dendrogram")
    plt.xlabel("Articles")
    plt.ylabel("Distance")
    st.pyplot(fig)

    st.info("Look for large vertical gaps → choose cluster count")

# ---------------- APPLY CLUSTERING ----------------
st.header("Apply Clustering")

num_clusters = st.slider("Number of clusters", 2, 10, 5)

if st.button("Apply Clustering"):

    hc = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric=distance_metric,
        linkage=linkage_method
    )

    labels = hc.fit_predict(X)

    # PCA for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Scatter plot
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_2d[:,0], X_2d[:,1], c=labels)
    plt.title("Cluster Visualization (PCA)")
    st.pyplot(fig)

    # Silhouette
    score = silhouette_score(X, labels)
    st.subheader("Silhouette Score")
    st.write(score)

    # ---------------- CLUSTER SUMMARY ----------------
    st.header("Cluster Summary")

    terms = vectorizer.get_feature_names_out()
    df_clusters = pd.DataFrame({"cluster": labels, "text": text_data})

    for c in range(num_clusters):
        st.subheader(f"Cluster {c}")
        cluster_texts = df_clusters[df_clusters.cluster==c]["text"]

        if len(cluster_texts) == 0:
            continue

        # top words
        tfidf_mean = X[labels==c].mean(axis=0)
        top_idx = np.argsort(tfidf_mean)[-10:]
        top_words = [terms[i] for i in top_idx]

        st.write("Top keywords:", top_words)
        st.write("Sample article:", cluster_texts.iloc[0][:200])

    # ---------------- BUSINESS INSIGHTS ----------------
    st.header("Business Insights")

    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for automatic tagging, recommendations, and content organization."
    )