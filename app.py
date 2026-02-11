import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")
st.title("🟣 News Topic Discovery with Hierarchical Clustering")
st.markdown("This system groups similar financial news articles automatically based on textual similarity.")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload Financial News Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Handle encoding safely
try:
    df = pd.read_csv(uploaded_file)
except:
    df = pd.read_csv(uploaded_file, encoding="latin1")

# Detect text column
text_cols = df.select_dtypes(include=["object"]).columns
if len(text_cols) == 0:
    st.error("No text column detected.")
    st.stop()

text_column = text_cols[1] if len(text_cols) > 1 else text_cols[0]

st.success(f"Loaded dataset: {uploaded_file.name}")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ===============================
# SIDEBAR CONFIG
# ===============================
st.sidebar.header("⚙ Configuration")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 1500, 500)
use_stopwords = st.sidebar.checkbox("Use English Stopwords", True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_option == "Unigrams":
    ngram_range = (1, 1)
elif ngram_option == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

num_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
dendro_subset = st.sidebar.slider("Articles for Dendrogram", 20, 150, 80)

# ===============================
# TF-IDF (Cached)
# ===============================
@st.cache_data
def compute_tfidf(text_data, max_features, use_stopwords, ngram_range):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if use_stopwords else None,
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer

# ===============================
# GENERATE DENDROGRAM
# ===============================
if st.button("Generate Dendrogram"):

    with st.spinner("Generating dendrogram..."):

        X, vectorizer = compute_tfidf(df[text_column], max_features, use_stopwords, ngram_range)

        n_comp = min(50, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_reduced = svd.fit_transform(X)

        subset = X_reduced[:dendro_subset]
        linked = linkage(subset, method=linkage_method)

        fig, ax = plt.subplots(figsize=(8, 4))
        dendrogram(linked, ax=ax)
        ax.set_title("Dendrogram")
        ax.set_ylabel("Distance")
        st.pyplot(fig)

# ===============================
# APPLY CLUSTERING
# ===============================
if st.button("Apply Clustering"):

    with st.spinner("Clustering in progress..."):

        X, vectorizer = compute_tfidf(df[text_column], max_features, use_stopwords, ngram_range)

        n_comp = min(50, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_reduced = svd.fit_transform(X)

        if linkage_method == "ward":
            model = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward")
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
        # PCA VISUALIZATION
        # ===============================
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_reduced)

        fig_scatter = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df["Cluster"].astype(str),
            hover_data=[df[text_column]],
            title="Cluster Visualization (PCA Projection)"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # ===============================
        # CLUSTER DISTRIBUTION
        # ===============================
        cluster_counts = df["Cluster"].value_counts().sort_index()

        fig_bar = go.Figure(
            data=[go.Bar(x=cluster_counts.index.astype(str),
                         y=cluster_counts.values)]
        )

        fig_bar.update_layout(
            title="Cluster Distribution",
            xaxis_title="Cluster ID",
            yaxis_title="Number of Articles"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # ===============================
        # CLUSTER SUMMARY
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

            sample_text = df[df["Cluster"] == i][text_column].iloc[0]

            summary.append({
                "Cluster ID": i,
                "Number of Articles": len(indices),
                "Top Keywords": ", ".join(top_words),
                "Sample Article": sample_text[:120] + "..."
            })

        st.dataframe(pd.DataFrame(summary))

        # ===============================
        # VALIDATION
        # ===============================
        st.subheader("Validation")
        st.metric("Silhouette Score", round(score, 4))

        if score > 0.5:
            st.success("Clusters are well separated.")
        elif score > 0.2:
            st.warning("Clusters have moderate overlap.")
        else:
            st.info("Clusters have some overlap.")

        # ===============================
        # EDITORIAL INSIGHTS
        # ===============================
        st.subheader("Editorial Insights")

        for row in summary:
            st.markdown(
                f"🟣 **Cluster {row['Cluster ID']}**: Articles focus on topics related to {row['Top Keywords']}."
            )
