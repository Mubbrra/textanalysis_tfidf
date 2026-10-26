"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Instructions:
Complete each TODO section below. You can run this file interactively cell-by-cell
in VS Code / PyCharm / Spyder (using the `# %%` markers) or directly via terminal:
    python lab15_starter.py
"""


# ## Setup & Dataset Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Load 4 real-world unstructured user discussion categories
CATEGORIES = [
    "sci.electronics",
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med",
]

print("Loading dataset...")
raw_data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

df = pd.DataFrame(
    {
        "text": raw_data.data,
        "true_label": [raw_data.target_names[i] for i in raw_data.target],
    }
)



# Keep posts with meaningful length (> 50 characters)
df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)
print(f"Loaded {len(df)} unstructured user posts.\n")


# %% [markdown]
# ## Step 1: Text Cleaning, Tokenization & Normalization (10 mins)

# %%




sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [
    lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens
]

# TODO 1.1: Tokenize `sample_post` using `word_tokenize`, convert to lowercase,
# and keep only alphabetic tokens (`w.isalpha()`).

# TODO 1.2: Remove standard English stopwords using `stopwords.words('english')`.


# TODO 1.3: Apply PorterStemmer and WordNetLemmatizer (with pos='v') to `filtered_tokens`.

# %% [markdown]
# ## Step 2: Bag of Words vs. TF-IDF Feature Extraction (15 mins)

# %%
# TODO 2.1: Initialize CountVectorizer and TfidfVectorizer with:
# stop_words='english', max_features=4000, min_df=5
bow_vec = None  # Replace None
tfidf_vec = None  # Replace None


bow_vec = CountVectorizer(
    stop_words="english", max_features=4000, min_df=5
)
tfidf_vec = TfidfVectorizer(
    stop_words="english", max_features=4000, min_df=5
)




# TODO 2.2: Fit and transform `df['text']` to create `X_bow` and `X_tfidf`
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: Calculate matrix sparsity (% of zero entries in `X_tfidf`)
if X_tfidf is not None:
    total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
    sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

    # TODO 2.4: Compute mean score per feature across documents and plot Top 10 TF-IDF words
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idx = tfidf_means.argsort()[-10:]
    top_words = tfidf_vec.get_feature_names_out()[top_idx]
    top_scores = tfidf_means[top_idx]

    plt.figure(figsize=(8, 4.5))
    plt.barh(top_words, top_scores, color="teal")
    plt.xlabel("Average TF-IDF Weight")
    plt.title("Top 10 Most Influential Words (Baseline TF-IDF)")
    plt.tight_layout()
    plt.savefig("top_features.png", dpi=150)
    plt.show()


# %% [markdown]
# ## Step 3: Automated Topic Discovery & Spatial Mapping (15 mins)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_tfidf)

    feature_names = tfidf_vec.get_feature_names_out()

    # Print the top 8 words for each cluster
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        top_indices = np.argsort(center)[::-1][:8]
        top_words = feature_names[top_indices]
        print(f"Cluster {cluster_id} top words: {', '.join(top_words)}")

    # Build a table of the top 5 words per cluster for the heatmap
    cluster_terms = {}
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        top_indices = np.argsort(center)[::-1][:5]
        cluster_terms[f"Cluster {cluster_id}"] = {
            feature_names[i]: center[i] for i in top_indices
        }

    heatmap_data = pd.DataFrame.from_dict(
        cluster_terms, orient="index"
    ).fillna(0)

    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Top TF-IDF Terms by Cluster")
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150)
    plt.show()

    # Project documents into 2D using TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    coordinates = svd.fit_transform(X_tfidf)

    plt.figure(figsize=(8, 6))
    points = plt.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=df["cluster"],
        cmap="tab10",
        s=15,
        alpha=0.7,
    )
    plt.colorbar(points, label="Cluster")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.title("2D Spatial Mapping of Document Clusters")
    plt.tight_layout()
    plt.savefig("spatial_mapping.png", dpi=150)
    plt.show()

# %%
if X_tfidf is not None:
    # TODO 3.1: Fit KMeans with `n_clusters=4, random_state=42, n_init=10` on `X_tfidf`
    kmeans = None  # Replace None
    # df["cluster"] = kmeans.fit_predict(X_tfidf)

    # TODO 3.2: Print the top 8 defining TF-IDF words per cluster centroid
    # and plot a seaborn heatmap (`cluster_heatmap.png`) of the top 5 words per cluster.

    # TODO 3.3: Project `X_tfidf` to 2 dimensions using `TruncatedSVD(n_components=2, random_state=42)`
    # and save a 2D scatter plot (`spatial_mapping.png`) colored by `df["cluster"]`.
    pass


# %% [markdown]
# ## Step 4: Problem-Solving Diagnostics (20 mins)

# %%
# Challenge 1 (Noise Filtering):
# TODO 4.1: Build a cleaner `TfidfVectorizer` using `token_pattern=r'\b[a-zA-Z]{3,}\b'`,
# `max_df=0.5`, and custom conversational stopwords (e.g., 'know', 'like', 'just', 'think', 'does', 'use', 'thanks').
# Re-fit KMeans and compare cluster keywords!
custom_stop_words = set(stopwords.words("english"))
custom_stop_words.update(
    {"know", "like", "just", "think", "does", "use", "thanks"}
)

clean_tfidf_vec = TfidfVectorizer(
    stop_words=sorted(custom_stop_words),
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    max_df=0.5,
    max_features=4000,
    min_df=5,
)

X_clean_tfidf = clean_tfidf_vec.fit_transform(df["text"])

clean_kmeans = KMeans(
    n_clusters=4, random_state=42, n_init=10
)
clean_clusters = clean_kmeans.fit_predict(X_clean_tfidf)

clean_features = clean_tfidf_vec.get_feature_names_out()

print("\nTop words after filtering conversational noise:")
for cluster_id, center in enumerate(clean_kmeans.cluster_centers_):
    top_indices = np.argsort(center)[::-1][:8]
    top_words = clean_features[top_indices]
    print(f"Cluster {cluster_id}: {', '.join(top_words)}")

# Challenge 2 (Hardware Overlap Diagnosis):
# TODO 4.2: Run `pd.crosstab(df['true_label'], df['cluster'])` and inspect overlapping categories.
overlap_table = pd.crosstab(df["true_label"], df["cluster"])
print("\nTrue categories compared with clusters:")
print(overlap_table)

# Estimate which two categories share clusters most
labels = list(overlap_table.index)
pair_overlap = {}

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        shared_count = np.minimum(
            overlap_table.loc[labels[i]].values,
            overlap_table.loc[labels[j]].values,
        ).sum()
        pair_overlap[(labels[i], labels[j])] = shared_count

most_overlapping = max(pair_overlap, key=pair_overlap.get)
print("\nCategories with the greatest cluster overlap:", most_overlapping)

# Show example posts whose category differs from their cluster's majority category
cluster_to_label = overlap_table.idxmax(axis=0)
df["cluster_majority_label"] = df["cluster"].map(cluster_to_label)
misclustered = df[df["true_label"] != df["cluster_majority_label"]]

print("\nExample posts to inspect:")
print(
    misclustered[["true_label", "cluster", "text"]]
    .head(5)
    .to_string(index=False)
)

# Challenge 3 (Rant Length Bias):
# TODO 4.3: Compare L2 vector norms (`np.linalg.norm`) of the longest and shortest posts in `X_bow` vs `X_tfidf`.
post_lengths = df["text"].str.len()
longest_idx = post_lengths.idxmax()
shortest_idx = post_lengths.idxmin()

print("\nLongest post length:", post_lengths[longest_idx])
print("Shortest post length:", post_lengths[shortest_idx])

bow_long_norm = np.linalg.norm(X_bow[longest_idx].toarray())
bow_short_norm = np.linalg.norm(X_bow[shortest_idx].toarray())

tfidf_long_norm = np.linalg.norm(X_tfidf[longest_idx].toarray())
tfidf_short_norm = np.linalg.norm(X_tfidf[shortest_idx].toarray())

print(f"\nBoW norm — longest post:  {bow_long_norm:.4f}")
print(f"BoW norm — shortest post: {bow_short_norm:.4f}")
print(f"TF-IDF norm — longest post:  {tfidf_long_norm:.4f}")
print(f"TF-IDF norm — shortest post: {tfidf_short_norm:.4f}")

# Challenge 4 (Stakeholder Explainability Audit):
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    """
    TODO 4.4: Return or print the top_n highest-weighted TF-IDF terms and their exact weights
    for a single document `doc_id` (test with doc_id=412).
    """
    row = matrix.getrow(doc_id).toarray().ravel()
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = np.flatnonzero(row)

    top_indices = nonzero_indices[
        np.argsort(row[nonzero_indices])[::-1][:top_n]
    ]

    result = [
        (feature_names[i], float(row[i]))
        for i in top_indices
    ]

    print(f"Document {doc_id} top terms: {result}")
    return result

explain_document(412, X_tfidf, tfidf_vec, top_n=5)







tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]