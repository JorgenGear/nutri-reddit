import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import os
import spacy

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import plotly.express as px
import praw

# ----------------------------- #
# 🔧 Streamlit Config
# ----------------------------- #
st.set_page_config(page_title="Reddit Nutrition Helper", layout="wide")

# ----------------------------- #
# 📦 Load Models
# ----------------------------- #
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    nlp = spacy.load("en_core_web_sm")
    return embedder, summarizer, nlp


embedder, summarizer, nlp = load_models()

# ----------------------------- #
# 🧽 Text Preprocessing
# ----------------------------- #
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"u/\w+|r/\w+", "", text)
    text = re.sub(r"\s+", " ", text).lower()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# ----------------------------- #
# 📂 Load Default Data
# ----------------------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("reddit_nutrition_data.csv")
    df.dropna(subset=["text"], inplace=True)
    df["text"] = df["title"].fillna('') + " " + df["text"].fillna('')
    df["cleaned_text"] = df["text"].apply(clean_text)
    return df

df = load_data()

# ----------------------------- #
# 🔐 Reddit API Client (PRAW)
# ----------------------------- #
@st.cache_resource
def get_reddit_client():
    creds = st.secrets["reddit"]
    reddit = praw.Reddit(
        client_id=creds["client_id"],
        client_secret=creds["client_secret"],
        user_agent=creds["user_agent"]
    )
    return reddit

def scrape_subreddit(subreddit_name, limit=500):
    reddit = get_reddit_client()
    posts = []
    try:
        for post in reddit.subreddit(subreddit_name).new(limit=limit):
            posts.append({
                "id": post.id,
                "subreddit": subreddit_name,
                "title": post.title,
                "text": post.selftext,
                "upvotes": post.score,
                "num_comments": post.num_comments,
                "timestamp": post.created_utc,
                "url": post.url
            })
        df_new = pd.DataFrame(posts)
        df_new["text"] = df_new["title"].fillna('') + " " + df_new["text"].fillna('')
        df_new["cleaned_text"] = df_new["text"].apply(clean_text)
        return df_new
    except Exception as e:
        st.error(f"❌ Failed to scrape r/{subreddit_name}: {e}")
        return None

# ----------------------------- #
# 🔮 Embedding & Clustering
# ----------------------------- #
@st.cache_resource
def embed_corpus(texts):
    return embedder.encode(texts, convert_to_tensor=True)

@st.cache_resource
def run_clustering(df, num_clusters=6):
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X = tfidf.fit_transform(df["cleaned_text"])
    km = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = km.fit_predict(X)

    terms = tfidf.get_feature_names_out()
    top_terms = {}
    for i in range(num_clusters):
        center = km.cluster_centers_[i]
        top = center.argsort()[-5:][::-1]
        top_terms[i] = ", ".join([terms[j] for j in top])
    return df, top_terms

# Default clustering
df, cluster_labels = run_clustering(df)
corpus_embeddings = embed_corpus(df["cleaned_text"].tolist())

# ----------------------------- #
# 🔍 Search + Summarize
# ----------------------------- #
def search(query, top_k=5):
    query_vec = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_vec, corpus_embeddings)[0]
    top_indices = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    return df.iloc[top_indices]

@st.cache_data(show_spinner=False)
def summarize_text(text):
    try:
        result = summarizer(text[:1024], max_length=120, min_length=30, do_sample=False)
        return result[0]["summary_text"]
    except:
        return "Summary failed or input too short."

# ----------------------------- #
# 🧭 Sidebar Controls
# ----------------------------- #
with st.sidebar:
    st.title("🧰 Controls")
    st.markdown("""
**ℹ️ How it works:**
- Loads ~1,500 Reddit posts on nutrition
- Lets you search & summarize them
- Optionally scrape your own subreddit
""")

    query = st.text_input("🔍 Ask a question:", value="healthy low-carb snacks")
    top_k = st.slider("How many results?", 3, 10, 5)
    sort_method = st.radio("Sort results by:", ["Relevance", "Upvotes", "Comments"])

    st.markdown("---")
    st.subheader("🌍 Load Custom Subreddit")
    custom_sub = st.text_input("Subreddit (no r/):", value="", placeholder="e.g. intermittentfasting")
    load_custom = st.button("Load Subreddit")

    if load_custom and custom_sub:
        with st.spinner(f"Scraping r/{custom_sub}..."):
            df_custom = scrape_subreddit(custom_sub)
            if df_custom is not None:
                with st.spinner("Embedding and clustering..."):
                    corpus_embeddings = embed_corpus(df_custom["cleaned_text"].tolist())
                    df_custom, cluster_labels = run_clustering(df_custom)
                    df = df_custom
                    st.success(f"✅ Loaded {len(df)} posts from r/{custom_sub}")

    st.markdown("---")
    st.subheader("🧠 Explore Topic Clusters")
    selected_cluster = st.selectbox("Choose cluster:", sorted(df["cluster"].unique()))
    st.markdown(f"**Top Keywords:** `{cluster_labels[selected_cluster]}`")

# ----------------------------- #
# 🎯 Main App Body
# ----------------------------- #
st.title("🥦 Reddit Nutrition Helper")
st.markdown("Get instant summaries & insights from Reddit users on nutrition, diet, and health topics.")

if query:
    st.subheader("🔍 Search Results")
    with st.spinner("Finding relevant posts..."):
        results = search(query, top_k=top_k)

        if sort_method == "Upvotes":
            results = results.sort_values("upvotes", ascending=False)
        elif sort_method == "Comments":
            results = results.sort_values("num_comments", ascending=False)

        for i, row in results.iterrows():
            with st.expander(f"📌 {row['title']} — ↑ {row['upvotes']} 💬 {row['num_comments']}"):
                st.markdown(f"**Subreddit:** `{row['subreddit']}`")
                st.markdown(f"[🔗 View on Reddit]({row['url']})", unsafe_allow_html=True)
                with st.spinner("Summarizing..."):
                    st.success(summarize_text(row["text"]))
                if st.checkbox("Show full post", key=f"post_{i}"):
                    st.info(row["text"])

        st.download_button(
            label="📥 Download results as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="search_results.csv",
            mime="text/csv"
        )

st.markdown("---")
st.subheader(f"🧵 Cluster {selected_cluster}: {cluster_labels[selected_cluster]}")
cluster_df = df[df["cluster"] == selected_cluster].sample(n=min(5, len(df[df["cluster"] == selected_cluster])))

for i, row in cluster_df.iterrows():
    with st.expander(f"🧠 {row['title']} — ↑ {row['upvotes']}"):
        st.write(row["text"])
        st.markdown(f"[🔗 View post]({row['url']})", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Post Distribution by Cluster")
cluster_counts = df["cluster"].value_counts().sort_index()
fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
             labels={'x': 'Cluster', 'y': 'Posts'},
             title="Number of Posts per Topic Cluster")
st.plotly_chart(fig)
