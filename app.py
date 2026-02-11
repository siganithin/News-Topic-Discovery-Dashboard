import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.write("Hierarchical Clustering for automatic topic discovery.")

# ===============================
# FILE UPLOAD
# ===============================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded_file, encoding="latin1")


text_cols = df.select_dtypes(include=["object"]).columns

if len(text_cols) == 0:
    st.error("No text column detected.")
    st.stop()

text_column = text_cols[0]
st.success(f"Detected text column: {text_column}")

# ===============================
# SIDEBAR SETTINGS
# ===============================

st.sidebar.header("Vectorization")

max_features = st.sidebar.slider("Max TF-IDF Features", 100, 1500, 500)
remove_stopwords = st.sidebar.checkbox("Remove Stopwords", True)

st.sidebar.header("Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

num_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
dendro_subset = st.sidebar.slider("Dendrogram Subset Size", 20, 100, 50)

# ===============================
# CACHED TF-IDF FUNCTION
# ===============================

@st.cache_data
def compute_tfidf(text_data, max_features, remove_stopwords):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if remove_stopwords else None
    )
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer

# ===============================
# DENDROGRAM BUTTON
# ===============================

if st.button("Generate Dendrogram"):

    with st.spinner("Generating dendrogram..."):

        X, vectorizer = compute_tfidf(df[text_column], max_features, remove_stopwords)

        svd_components = min(50, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        X_reduced = svd.fit_transform(X)

        subset = X_reduced[:dendro_subset]
        linked = linkage(subset, method=linkage_method)

        fig, ax = plt.subplots(figsize=(8, 4))
        dendrogram(linked, ax=ax)
        ax.set_ylabel("Distance")
        st.pyplot(fig)

# ===============================
# CLUSTERING BUTTON
# ===============================

if st.button("Apply Clustering"):

    with st.spinner("Clustering in progress..."):

        X, vectorizer = compute_tfidf(df[text_column], max_features, remove_stopwords)

        svd_components = min(50, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        X_reduced = svd.fit_transform(X)

        if linkage_method == "ward":
            model = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage="ward"
            )
            labels = model.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, labels)
        else:
            model = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=linkage_method,
                metric="cosine"
            )
            labels = model.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, labels, metric="cosine")

        df["Cluster"] = labels

        # ===============================
        # PCA Visualization
        # ===============================

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_reduced)

        fig2 = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df["Cluster"].astype(str),
            hover_data=[df[text_column]],
            title="Cluster Visualization"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ===============================
        # Silhouette Score
        # ===============================

        st.subheader("Silhouette Score")
        st.write(round(score, 4))

        # ===============================
        # Cluster Summary
        # ===============================

        st.subheader("Cluster Summary")

        feature_names = vectorizer.get_feature_names_out()
        summary = []

        for i in range(num_clusters):
            indices = np.where(labels == i)[0]
            cluster_mean = X[indices].mean(axis=0)
            cluster_mean = np.array(cluster_mean).flatten()
            top_indices = cluster_mean.argsort()[-10:][::-1]
            top_words = [feature_names[j] for j in top_indices]

            summary.append({
                "Cluster": i,
                "Articles": len(indices),
                "Top Keywords": ", ".join(top_words)
            })

        st.dataframe(pd.DataFrame(summary))