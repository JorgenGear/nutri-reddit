🥦 Reddit Nutrition Helper
An intelligent Streamlit app that summarizes and clusters nutrition-related Reddit threads using advanced NLP techniques.

💡 Overview
Reddit Nutrition Helper is a lightweight, interactive web app that:

🔍 Lets users search Reddit posts semantically (not just keywords)

🧾 Summarizes long Reddit posts into short AI-generated takeaways

🧠 Clusters posts into topic groups using TF-IDF + KMeans

🌍 Lets you scrape any subreddit in real-time and analyze it instantly

📸 Screenshots

⚙️ Features

Feature	Description
🔍 Semantic Search	Uses transformer embeddings (SBERT) to find relevant Reddit posts
🧾 Text Summarization	Generates concise summaries using BART transformer
🧠 Topic Clustering	Groups posts by topic using TF-IDF and KMeans
🌍 Custom Subreddit Loader	Type in any subreddit and instantly scrape + analyze the latest 500 posts
📊 Visualization	Bar chart shows cluster distributions
📥 CSV Export	Download search results for your own analysis
📁 Files & Structure
text
Copy
📦 reddit-nutrition-helper
├── app.py                   # Streamlit app entrypoint
├── reddit_nutrition_data.csv  # Default corpus from r/nutrition, r/diet, r/loseit
├── .gitignore
├── requirements.txt
└── .streamlit/
    └── secrets.toml         # (⚠️ NOT committed – used locally only)
🧠 Methods Used

Unit	Technique
Unit 1	NLP Preprocessing, Corpus Collection via Reddit API
Unit 3	Text Similarity, Summarization, Clustering
Unit 4	Pretrained Transformers (BART, SBERT)
🚀 Deployment
This app can be deployed on Streamlit Cloud:

Push this repo to GitHub (excluding .streamlit/secrets.toml)

Go to Streamlit Cloud > "New app"

Set app.py as the app file

Add your Reddit API credentials under “Secrets” as:

toml
Copy
[reddit]
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
user_agent = "reddit_nutrition_app"
🔐 Reddit API Setup (for local use)
Go to Reddit Developer Console

Create a new app → choose script

Add client_id, client_secret, user_agent to .streamlit/secrets.toml (do not commit this file!)

📦 Install & Run Locally
bash
Copy
# Create and activate a virtual environment
pip install -r requirements.txt
streamlit run app.py
🙋 Author
Rhett Jorgensen
GitHub
Spring 2025 — DATA 6420

📜 License
MIT License. Use, remix, and deploy responsibly.