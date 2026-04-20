# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Phase 2 — Flask REST API for Topic Prediction
# ============================================
#
# Serves BERTopic model predictions via REST API.
# Run: python3 api_server.py
# Endpoints:
#   POST /api/predict        — classify a new article
#   GET  /api/topics         — list all discovered topics
#   GET  /api/plots/<name>   — return base64 plot image
#   GET  /api/summary        — Phase 1 & 2 summary stats
#   GET  /api/coherence      — per-topic embedding coherence
#   GET  /api/separation     — ACA vs COVID context separation scores
#   GET  /api/purity         — per-topic category purity scores
#   GET  /api/attribution/<topic_id> — token attribution rows for a topic
#   POST /api/attribution_live      — live token attribution for input text
#   POST /api/similarity     — compare BoW vs SBERT cosine for two texts
#
# Model caching:
#   On first run the fitted model and embeddings are saved
#   to bert_plots/cached_model/ so subsequent starts load
#   in ~10 seconds.  Delete the cache folder to force a
#   full refit.
# ============================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
import numpy as np
import json
import os
import re
import base64
import pickle

# ============================================
# GUNICORN / MULTI-PROCESS SAFETY
# With preload=True (set in entrypoint.sh), the model is loaded
# once in the master process and forked to workers — no duplicate
# loading. TOKENIZERS_PARALLELISM=false prevents HuggingFace
# tokenizer deadlocks in forked workers.
# ============================================
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

app   = Flask(__name__)
CORS(app)

CACHE_DIR = 'bert_plots/cached_model'
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================
# HELPER: text cleaning (same as bert_model.py)
# ============================================

def clean_text(text):
    """Minimal cleaning preserving SBERT-compatible surface form.
    Retains digits (unlike lda_model.py) because numbers like '19'
    in 'covid 19' carry discriminative temporal meaning for SBERT.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_for_attribution(text):
    """Tokenize text in the same cleaned space used for SBERT input."""
    cleaned = clean_text(text)
    toks = [t for t in cleaned.split(' ') if t]
    return toks


def compute_live_token_attribution(text):
    """LIME-style perturbation attribution on a single input text.

    attribution(i) = 1 - cosine( e(text), e(text without token_i) )
    """
    tokens = tokenize_for_attribution(text)
    if not tokens:
        return []
    if len(tokens) > 60:
        tokens = tokens[:60]

    original_text = ' '.join(tokens)
    e_original = embedding_model.encode([original_text])[0]
    e_original = normalize(np.array([e_original]))[0]

    rows = []
    for i, tok in enumerate(tokens):
        masked_tokens = tokens[:i] + tokens[i+1:]
        masked_text = ' '.join(masked_tokens).strip()
        if not masked_text:
            score = 1.0
        else:
            e_masked = embedding_model.encode([masked_text])[0]
            e_masked = normalize(np.array([e_masked]))[0]
            score = 1.0 - float(np.dot(e_original, e_masked))
        rows.append({
            'position': i,
            'token': tok,
            'attribution_score': round(float(score), 6)
        })

    rows.sort(key=lambda x: x['attribution_score'], reverse=True)
    for rnk, row in enumerate(rows, start=1):
        row['rank'] = rnk
    rows.sort(key=lambda x: x['position'])
    return rows

# ============================================
# STARTUP: LOAD OR BUILD MODEL
# ============================================

print("=" * 60)
print("DYNAMIC TREND & EVENT DETECTOR — API Server")
print("=" * 60)

print("\nLoading dataset...")
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
df = df[['headline', 'short_description', 'category', 'date']].copy()
df['text'] = df['headline'] + ' ' + df['short_description']
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)
df = df.sort_values('date').reset_index(drop=True)
df = df.groupby(df['date'].dt.year).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)
df['clean_text'] = df['text'].apply(clean_text)
docs = df['clean_text'].tolist()

print("\nLoading SBERT model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings_path   = os.path.join(CACHE_DIR, 'embeddings.npy')
bertopic_path     = os.path.join(CACHE_DIR, 'bertopic_model')
bow_vec_path      = os.path.join(CACHE_DIR, 'bow_vec.pkl')

if (os.path.exists(embeddings_path) and
        os.path.exists(bertopic_path) and
        os.path.exists(bow_vec_path)):
    print("\nLoading cached model (delete bert_plots/cached_model/ to refit)...")
    embeddings  = np.load(embeddings_path)
    topic_model = BERTopic.load(bertopic_path)
    with open(bow_vec_path, 'rb') as f:
        bow_vec = pickle.load(f)
    topics = topic_model.transform(docs, embeddings)[0]
    df['bert_topic'] = topics
    print("Cached model loaded.")

else:
    print("\nNo cache found — fitting BERTopic (~2 minutes)...")
    print("(Subsequent starts will use the saved cache)\n")

    print("  Encoding documents with SBERT...")
    embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=64)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,      # 5-dim: preserves structure for HDBSCAN without
            min_dist=0.0,        # excessive runtime cost (default per Grootendorst 2022)
            metric='cosine',     # cosine: aligns with SBERT's training objective
            random_state=42
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric='euclidean',
            prediction_data=True   # required for .transform() on new docs
        ),
        vectorizer_model=CountVectorizer(
            stop_words='english',
            min_df=5,            # removes hapax legomena from 29-token avg docs
            max_df=0.95,         # removes near-universal terms
            ngram_range=(1, 2)   # bigrams for compound terms: 'covid 19', 'health care'
        ),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        nr_topics='auto',
        top_n_words=10,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)

    # Reduce outliers using c-TF-IDF assignment strategy
    print("  Reducing outliers via c-TF-IDF strategy (threshold=0.1)...")
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
    topic_model.update_topics(docs, topics=new_topics)
    topics = new_topics
    df['bert_topic'] = topics

    # Build BoW vectorizer for cross-representation similarity
    bow_vec = CountVectorizer(
        max_features=5000,
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    bow_vec.fit(docs)

    np.save(embeddings_path, embeddings)
    topic_model.save(bertopic_path)
    with open(bow_vec_path, 'wb') as f:
        pickle.dump(bow_vec, f)
    print(f"\nModel cached to {CACHE_DIR}/")

# Build topic info cache
topic_info_cache = {}
for _, row in topic_model.get_topic_info().iterrows():
    tid = row['Topic']
    if tid == -1:
        topic_info_cache[-1] = {
            'id': -1, 'name': 'Outlier / Unclassified',
            'words': [], 'count': int(row['Count'])
        }
        continue
    words = topic_model.get_topic(tid)
    topic_info_cache[tid] = {
        'id':    int(tid),
        'name':  ' / '.join([w for w, _ in words[:3]]),
        'words': [{'word': w, 'score': round(float(s), 4)}
                  for w, s in words[:10]],
        'count': int(row['Count'])
    }

n_topics   = len([k for k in topic_info_cache if k != -1])
n_outliers = int((df['bert_topic'] == -1).sum())

print(f"\nAPI ready — {n_topics} topics, {len(df):,} documents indexed.")
print(f"Outlier documents: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
print("="*60)

# ============================================
# HELPER FUNCTIONS
# ============================================

def encode_image(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_bow_similarity(text_a: str, text_b: str) -> float:
    """Return BoW cosine similarity between two texts."""
    vecs = bow_vec.transform([text_a, text_b])
    n    = normalize(vecs.toarray(), axis=1)
    return float(n[0] @ n[1])


def get_sbert_similarity(text_a: str, text_b: str) -> float:
    """Return SBERT cosine similarity between two texts."""
    embs = embedding_model.encode([text_a, text_b])
    return float(cosine_similarity(embs)[0, 1])

# ============================================
# ENDPOINTS
# ============================================

@app.route('/api/predict', methods=['POST'])
def predict_topic():
    """Classify a new article into a BERTopic topic.

    Expects JSON: { "text": "article text here" }
    Returns:
      - predicted topic (id, name, top words, c-TF-IDF scores)
      - confidence score
      - top-5 similar corpus articles (by SBERT cosine)
      - BoW vs SBERT similarity for the top match
    """
    data = request.json or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    emb     = embedding_model.encode([cleaned])
    topic, prob = topic_model.transform([cleaned], emb)

    tid        = int(topic[0])
    confidence = float(prob[0].max()) if hasattr(prob[0], 'max') else float(prob[0])
    info       = topic_info_cache.get(tid, topic_info_cache.get(-1))

    # Top-5 similar corpus articles
    sims    = cosine_similarity(emb, embeddings)[0]
    top_idx = sims.argsort()[-5:][::-1]
    similar = []
    for idx in top_idx:
        sim_text = df.iloc[idx]['clean_text']
        similar.append({
            'headline':         df.iloc[idx]['headline'],
            'category':         df.iloc[idx]['category'],
            'date':             str(df.iloc[idx]['date'].date()),
            'sbert_similarity': round(float(sims[idx]), 4),
            'bow_similarity':   round(get_bow_similarity(cleaned, sim_text), 4),
            'topic':            int(df.iloc[idx]['bert_topic'])
        })

    return jsonify({
        'topic':            info,
        'confidence':       round(confidence, 4),
        'similar_articles': similar,
        'input_text':       text
    })


@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Return all discovered topics sorted by size."""
    topics_list = [v for k, v in topic_info_cache.items() if k != -1]
    topics_list.sort(key=lambda x: x['count'], reverse=True)
    return jsonify({'topics': topics_list, 'total': len(topics_list)})


@app.route('/api/plots/<name>', methods=['GET'])
def get_plot(name):
    """Return a base64-encoded plot image by filename."""
    safe_name = os.path.basename(name)
    for folder in ['bert_plots', 'model_plots', 'eda_plots']:
        path = os.path.join(folder, safe_name)
        if os.path.exists(path):
            return jsonify({'image': encode_image(path), 'name': safe_name})
    return jsonify({'error': f"Plot '{safe_name}' not found"}), 404


@app.route('/api/similarity', methods=['POST'])
def compare_representations():
    """Compare BoW vs SBERT cosine similarity for two texts.

    Demonstrates the context-separation difference between
    Phase 1 (BoW) and Phase 2 (SBERT) representations.

    Expects JSON: { "text_a": "...", "text_b": "..." }
    """
    data   = request.json or {}
    text_a = clean_text(data.get('text_a', ''))
    text_b = clean_text(data.get('text_b', ''))
    if not text_a or not text_b:
        return jsonify({'error': 'Provide text_a and text_b'}), 400

    bow_sim   = get_bow_similarity(text_a, text_b)
    sbert_sim = get_sbert_similarity(text_a, text_b)

    return jsonify({
        'text_a':          text_a,
        'text_b':          text_b,
        'bow_cosine':      round(bow_sim, 4),
        'sbert_cosine':    round(sbert_sim, 4),
        'separation_gain': round(bow_sim - sbert_sim, 4),
        'interpretation': (
            'SBERT separates these semantically distinct texts more '
            f'than BoW (gap = {bow_sim - sbert_sim:+.3f}). '
            'Positive gap means BoW incorrectly rates them as more '
            'similar than their actual semantics warrant.'
        )
    })


@app.route('/api/coherence', methods=['GET'])
def get_coherence():
    """Return per-topic embedding coherence scores.

    Note: coherence here is the mean intra-topic SBERT cosine similarity
    (embedding-space cluster tightness), NOT the standard c_v or u_mass
    word-level coherence metric used in NLP literature.
    """
    path = 'bert_plots/topic_coherence.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Coherence file not found — run bert_model.py first'}), 404
    cdf = pd.read_csv(path)
    return jsonify({
        'topics':       cdf.to_dict(orient='records'),
        'mean':         round(float(cdf['coherence'].mean()), 4),
        'min':          round(float(cdf['coherence'].min()), 4),
        'max':          round(float(cdf['coherence'].max()), 4),
        'metric_note':  'Embedding-space proxy: mean intra-topic SBERT cosine similarity.'
                        ' Not equivalent to c_v or u_mass word-level coherence.'
    })


@app.route('/api/separation', methods=['GET'])
def get_separation():
    """Return the ACA vs COVID context separation scores (Table 3 in report)."""
    path = 'bert_plots/context_separation_scores.json'
    if not os.path.exists(path):
        return jsonify({'error': 'Separation scores not found — run bert_model.py first'}), 404
    with open(path) as f:
        scores = json.load(f)

    bow   = scores.get('bow', {})
    sbert = scores.get('sbert', {})
    return jsonify({
        'bow':   bow,
        'sbert': sbert,
        'separation_gap': {
            'bow':   round(
                (bow.get('intra_aca', 0) + bow.get('intra_covid', 0)) / 2
                - bow.get('cross', 0), 4),
            'sbert': round(
                (sbert.get('intra_aca', 0) + sbert.get('intra_covid', 0)) / 2
                - sbert.get('cross', 0), 4),
        },
        'interpretation': (
            'A larger separation gap means the representation more cleanly '
            'distinguishes ACA legal documents from COVID pandemic documents '
            'despite shared vocabulary. BoW gap ≈ 0 (cannot separate); '
            'SBERT gap > 0.1 (clear separation).'
        )
    })


@app.route('/api/purity', methods=['GET'])
def get_purity():
    """Return per-topic category purity scores for BERTopic (Phase 2).

    Purity(k) = fraction of topic k documents belonging to the single
    most dominant HuffPost editorial category. Mirrors the LDA purity
    computed in lda_model.py Step 12 for direct comparison.

    Note: BERTopic purity may be lower than LDA purity for cross-category
    topics — this reflects correct semantic grouping across editorial
    boundaries, not a modeling failure.
    """
    path = 'bert_plots/bert_topic_purity.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Purity file not found — run bert_model.py first'}), 404
    pdf = pd.read_csv(path)
    return jsonify({
        'topics':         pdf.to_dict(orient='records'),
        'mean':           round(float(pdf['purity'].mean()), 4),
        'min':            round(float(pdf['purity'].min()), 4),
        'max':            round(float(pdf['purity'].max()), 4),
        'interpretation': (
            'Purity = fraction of topic documents in the single most dominant '
            'news category. Lower BERTopic purity vs LDA may indicate correct '
            'semantic grouping across editorial category boundaries.'
        )
    })


@app.route('/api/attribution/<topic_id>', methods=['GET'])
def get_attribution(topic_id):
    """Return token attribution details for the requested topic.

    topic_id can be an integer topic id or 'all'.
    Data source: bert_plots/token_attribution.csv produced by bert_model.py Step 13.
    """
    path = 'bert_plots/token_attribution.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Token attribution file not found — run bert_model.py first'}), 404

    adf = pd.read_csv(path)
    if adf.empty:
        return jsonify({
            'topic_id': topic_id,
            'rows': [],
            'top_tokens': [],
            'count': 0
        })

    if topic_id == 'all':
        subset = adf.copy()
    else:
        try:
            tid = int(topic_id)
        except ValueError:
            return jsonify({'error': 'topic_id must be an integer or "all"'}), 400
        subset = adf[adf['topic_id'] == tid].copy()

    if subset.empty:
        return jsonify({
            'topic_id': topic_id,
            'rows': [],
            'top_tokens': [],
            'count': 0
        })

    top_tokens_df = (
        subset.groupby('token', as_index=False)['attribution_score']
        .mean()
        .sort_values('attribution_score', ascending=False)
        .head(5)
    )
    top_tokens = [{
        'token': row['token'],
        'mean_attribution': round(float(row['attribution_score']), 6)
    } for _, row in top_tokens_df.iterrows()]

    return jsonify({
        'topic_id': topic_id,
        'count': int(len(subset)),
        'top_tokens': top_tokens,
        'rows': subset.to_dict(orient='records'),
        'interpretation': (
            'Token attribution measures how much removing each token changes the '
            'SBERT sentence embedding. Higher score means stronger contribution.'
        )
    })


@app.route('/api/attribution_live', methods=['POST'])
def get_live_attribution():
    """Return token attribution for a user-provided input text.

    Expects JSON: { "text": "..." }
    Returns token-wise attribution scores for dynamic interpretability UI.
    """
    data = request.json or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    rows = compute_live_token_attribution(text)
    if not rows:
        return jsonify({
            'input_text': text,
            'rows': [],
            'top_tokens': [],
            'count': 0
        })

    top_tokens = sorted(rows, key=lambda x: x['attribution_score'], reverse=True)[:5]
    return jsonify({
        'input_text': text,
        'count': len(rows),
        'rows': rows,
        'top_tokens': [{
            'token': t['token'],
            'attribution_score': t['attribution_score'],
            'rank': t['rank']
        } for t in top_tokens],
        'interpretation': (
            'Dynamic token attribution for this input text: higher scores indicate '
            'tokens whose removal causes larger SBERT embedding drift.'
        )
    })


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Return project summary statistics for both phases."""
    return jsonify({
        'phase1': {
            'method':          'LDA (Latent Dirichlet Allocation)',
            'paper':           'Blei, Ng, Jordan (2003) — JMLR 3:993-1022',
            'topics':          10,
            'confidence':      0.6103,
            'features':        ['Temporal Weight', 'Category Velocity', 'Text Richness'],
            'limitation':      'Bag-of-words conflation: ACA 2014 merged with COVID 2020'
        },
        'phase2': {
            'method':          'BERTopic (SBERT + UMAP + HDBSCAN + c-TF-IDF)',
            'paper':           'Grootendorst (2022) — arXiv:2203.05794',
            'sbert_paper':     'Reimers & Gurevych (2019) — arXiv:1908.10084',
            'umap_paper':      'McInnes, Healy, Melville (2018) — arXiv:1802.03426',
            'topics':          n_topics,
            'documents':       len(df),
            'outliers':        n_outliers,
            'outlier_rate':    round(n_outliers / len(df), 4),
            'outlier_strategy':'reduce_outliers(strategy=c-tf-idf, threshold=0.1)',
            'context_separation': True,
            # API retrieval/similarity uses cached base SBERT embeddings (384-dim).
            # TPI augmentation (32-dim) is used in training-time manifold shaping.
            'sbert_dim':         384,
            'tpi_dim':           32,
            'tpi_augmented_dim': 416,
            'retrieval_dim':     384,
            'reduction_dim':   5,
            'ngram_range':     '(1,2) — bigrams for compound domain terms',
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
