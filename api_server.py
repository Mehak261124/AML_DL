# ============================================
# Phase 2 — API Server for Topic Prediction
# Serves BERTopic model predictions via REST API
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import pandas as pd
import numpy as np
import json
import os
import re
import base64

app = Flask(__name__)
CORS(app)

# ---- Load model & data on startup ----
print("Loading SBERT model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading dataset for context...")
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
df = df[['headline', 'short_description', 'category', 'date']].copy()
df['text'] = df['headline'] + ' ' + df['short_description']
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)
df = df.sort_values('date').reset_index(drop=True)
df = df.groupby(df['date'].dt.year).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
docs = df['clean_text'].tolist()

print("Fitting BERTopic model (this takes ~2 min)...")
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean', prediction_data=True),
    vectorizer_model=CountVectorizer(stop_words='english', min_df=5, max_df=0.95, ngram_range=(1, 2)),
    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
    nr_topics='auto',
    top_n_words=10,
    verbose=False
)
topics, probs = topic_model.fit_transform(docs, embeddings)
df['bert_topic'] = topics

# Build topic info cache
topic_info_cache = {}
for _, row in topic_model.get_topic_info().iterrows():
    tid = row['Topic']
    if tid == -1:
        topic_info_cache[-1] = {'id': -1, 'name': 'Outlier', 'words': [], 'count': int(row['Count'])}
        continue
    words = topic_model.get_topic(tid)
    topic_info_cache[tid] = {
        'id': int(tid),
        'name': ' / '.join([w for w, _ in words[:3]]),
        'words': [{'word': w, 'score': round(float(s), 4)} for w, s in words[:10]],
        'count': int(row['Count'])
    }

print(f"API ready! {len(topic_info_cache)-1} topics loaded.")


def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


@app.route('/api/predict', methods=['POST'])
def predict_topic():
    """Predict topic for user-provided article text."""
    data = request.json
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    emb = embedding_model.encode([cleaned])
    topic, prob = topic_model.transform([cleaned], emb)

    tid = int(topic[0])
    confidence = float(prob[0].max()) if hasattr(prob[0], 'max') else float(prob[0])
    info = topic_info_cache.get(tid, topic_info_cache.get(-1))

    # Find similar articles from dataset
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(emb, embeddings)[0]
    top_indices = sims.argsort()[-5:][::-1]
    similar = []
    for idx in top_indices:
        similar.append({
            'headline': df.iloc[idx]['headline'],
            'category': df.iloc[idx]['category'],
            'date': str(df.iloc[idx]['date'].date()),
            'similarity': round(float(sims[idx]), 4),
            'topic': int(df.iloc[idx]['bert_topic'])
        })

    return jsonify({
        'topic': info,
        'confidence': round(confidence, 4),
        'similar_articles': similar,
        'input_text': text
    })


@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Return all discovered topics."""
    topics_list = [v for k, v in topic_info_cache.items() if k != -1]
    topics_list.sort(key=lambda x: x['count'], reverse=True)
    return jsonify({'topics': topics_list, 'total': len(topics_list)})


@app.route('/api/plots/<name>', methods=['GET'])
def get_plot(name):
    """Return base64-encoded plot image."""
    # Check both directories
    for folder in ['bert_plots', 'model_plots', 'eda_plots']:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return jsonify({'image': encode_image(path), 'name': name})
    return jsonify({'error': 'Plot not found'}), 404


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Return project summary stats."""
    n_topics = len([k for k in topic_info_cache if k != -1])
    n_outliers = int((df['bert_topic'] == -1).sum())
    return jsonify({
        'phase1': {
            'method': 'LDA (Latent Dirichlet Allocation)',
            'topics': 10,
            'confidence': 0.6103,
            'limitation': 'Bag-of-words conflation'
        },
        'phase2': {
            'method': 'BERTopic (SBERT + UMAP + HDBSCAN)',
            'topics': n_topics,
            'documents': len(df),
            'outliers': n_outliers,
            'context_separation': True
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
