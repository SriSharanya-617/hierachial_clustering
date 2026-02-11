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

# ---------------- TITLE ----------------
st.title("🟣 News Topic Discovery Dashboard")
st.caption("Hierarchical Clustering for automatic news grouping")

# ---------------- DATA ----------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")
else:
    st.info("Using default dataset preview mode")
    st.stop()

# Detect text column
text_col = st.selectbox("Select text column", df.columns)
text_data = df[text_col].dropna()

# ---------------- CONTROL PANEL ----------------
st.markdown("### Controls")

c1, c2, c3, c4 = st.columns(4)

with c1:
    max_features = st.slider("TF-IDF Features", 100, 2000, 1000)

with c2:
    stop_words = st.checkbox("Remove Stopwords", True)

with c3:
    linkage_method = st.selectbox("Linkage", ["ward","complete","average","single"])

with c4:
    dendro_size = st.slider("Dendrogram Sample", 20, 200, 100)

ngram_choice = st.selectbox("N-grams", ["Unigrams","Bigrams","Uni+Bigrams"])

if ngram_choice == "Unigrams":
    ngram_range = (1,1)
elif ngram_choice == "Bigrams":
    ngram_range = (2,2)
else:
    ngram_range = (1,2)

# ---------------- TFIDF ----------------
vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if stop_words else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(text_data).toarray()

# ---------------- DENDROGRAM ----------------
st.markdown("### 🌳 Dendrogram")

if st.button("Generate Dendrogram"):
    fig, ax = plt.subplots(figsize=(6,4))
    sch.dendrogram(sch.linkage(X[:dendro_size], method=linkage_method))
    plt.title("Dendrogram")
    st.pyplot(fig)

# ---------------- CLUSTER APPLY ----------------
st.markdown("### Apply Clustering")

num_clusters = st.slider("Number of clusters", 2, 10, 5)

if st.button("Apply Clustering"):

    hc = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric="euclidean",
        linkage=linkage_method
    )

    labels = hc.fit_predict(X)

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    col1, col2 = st.columns(2)

    # Scatter
    with col1:
        st.subheader("Cluster Visualization")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.scatter(X_2d[:,0], X_2d[:,1], c=labels)
        st.pyplot(fig2)

    # Silhouette
    with col2:
        st.subheader("Silhouette Score")
        score = silhouette_score(X, labels)
        st.metric("Score", round(score,3))

    # ---------------- SUMMARY ----------------
    st.markdown("### Topic Summary")

    terms = vectorizer.get_feature_names_out()

    for c in range(num_clusters):
        cluster_texts = text_data[labels==c]

        if len(cluster_texts)==0:
            continue

        tfidf_mean = X[labels==c].mean(axis=0)
        top_idx = np.argsort(tfidf_mean)[-5:]
        top_words = [terms[i] for i in top_idx]

        st.write(f"**Cluster {c}:**", ", ".join(top_words))
        st.caption(cluster_texts.iloc[0][:150])

    # ---------------- BUSINESS INSIGHT ----------------
    st.markdown("---")
    st.info(
        "Articles grouped in the same cluster share similar vocabulary and themes. "
        "These clusters can be used for automatic tagging, recommendations, and content organization."
    )