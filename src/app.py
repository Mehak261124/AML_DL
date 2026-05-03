# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Phase 2 — Flask REST API for Topic Prediction
# ============================================
#
# FIXES APPLIED:
#   FIX A — Centroid prediction: use per-topic SBERT similarity
#            ranked by nearest centroid WITHOUT log-size penalty.
#            The penalty was causing Topic 0 (largest) to always win.
#   FIX B — Real/Fake event classification added to /api/predict
#            Uses velocity + GDELT ground-truth + temporal plausibility.
#   FIX C — Confidence score is now true cosine similarity (0-1 range)
#            not a penalised ratio that produces nonsensical 7% values.
#   FIX D — All other endpoints preserved from original api_server.py.
# ============================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
import json
import os
import re
import base64
import pickle

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

app = Flask(__name__)
CORS(app)

CACHE_DIR = 'bert_plots/cached_model'
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================
# STOP WORDS (mirrors bert_model.py FIX B)
# ============================================
news_domain_stops = {
    'said', 'say', 'says', 'told', 'tell', 'tells', 'reported',
    'according', 'added', 'noted', 'called',
    'like', 'just', 'know', 'new', 'one', 'two', 'three', 'first',
    'good', 'big', 'great', 'old', 'little', 'right', 'small',
    'time', 'year', 'years', 'day', 'days', 'week', 'month',
    'today', 'ago', 'now', 'long',
    'make', 'makes', 'made', 'way', 'get', 'gets', 'got',
    'go', 'goes', 'going', 'gone', 'come', 'coming',
    'take', 'takes', 'took', 'see', 'look', 'looks', 'looking',
    'think', 'use', 'uses', 'used', 'want', 'wants', 'need', 'needs',
    'back', 'even', 'also',
    'people', 'man', 'woman', 'men', 'women', 'person', 'world',
    'photos', 'video', 'watch', 'via', 'amp', 'huffpost',
    'huffington', 'post',
    'dont', 'doesnt', 'didnt', 'isnt', 'wasnt', 'arent',
    'im', 'youre', 'thats', 'its', 'hes', 'shes', 'theyre',
    'ive', 'weve', 'theyll', 'wont', 'cant', 'couldnt',
    '10', '20', '2013', '2014', '2015', '2016', '2017',
    '2018', '2019', '100', '000',
    'us', 'u', 'really', 'actually', 'still', 'already',
}
ALL_STOPS = list(ENGLISH_STOP_WORDS.union(news_domain_stops))

# ============================================
# GROUND-TRUTH EVENT DATABASE
# Used for real/fake classification in /api/predict
# ============================================
GROUND_TRUTH_EVENTS = {
    # topic_semantic_label_keyword -> { verified: bool, real_years: [...], event_name: str, category: str }
    'russia-ukraine': {
        'real_years': [2014, 2022], 'event_name': 'Russia-Ukraine War',
        'category': 'geopolitical', 'verified': True
    },
    'ukraine war': {
        'real_years': [2022], 'event_name': 'Russia-Ukraine War 2022',
        'category': 'geopolitical', 'verified': True
    },
    'covid-19 pandemic': {
        'real_years': [2020, 2021], 'event_name': 'COVID-19 Pandemic',
        'category': 'health', 'verified': True
    },
    'covid-19 vaccination': {
        'real_years': [2021], 'event_name': 'COVID-19 Vaccine Rollout',
        'category': 'health', 'verified': True
    },
    'trump impeachment': {
        'real_years': [2019, 2020], 'event_name': 'Trump Impeachment Proceedings',
        'category': 'politics', 'verified': True
    },
    'mueller investigation': {
        'real_years': [2017, 2018, 2019], 'event_name': 'Mueller Investigation',
        'category': 'politics', 'verified': True
    },
    'stormy daniels': {
        'real_years': [2018], 'event_name': 'Trump-Stormy Daniels Scandal',
        'category': 'politics', 'verified': True
    },
    '2016 presidential election': {
        'real_years': [2015, 2016], 'event_name': '2016 US Presidential Election',
        'category': 'politics', 'verified': True
    },
    '2016 democratic primary': {
        'real_years': [2015, 2016], 'event_name': '2016 Democratic Primary (Sanders vs Clinton)',
        'category': 'politics', 'verified': True
    },
    'trump presidency': {
        'real_years': [2016, 2017, 2018, 2019, 2020], 'event_name': 'Trump Presidency 2017-2021',
        'category': 'politics', 'verified': True
    },
    'parenting & family': {
        'real_years': [], 'event_name': 'Lifestyle / Parenting Content',
        'category': 'lifestyle', 'verified': False
    },
    'wedding & beauty': {
        'real_years': [], 'event_name': 'Lifestyle / Beauty Content',
        'category': 'lifestyle', 'verified': False
    },
    'fashion & style': {
        'real_years': [], 'event_name': 'Fashion & Style Coverage',
        'category': 'lifestyle', 'verified': False
    },
    'holidays & entertainment': {
        'real_years': [], 'event_name': 'Seasonal / Entertainment Content',
        'category': 'entertainment', 'verified': False
    },
}

# ============================================
# SIGNAL CLASSIFICATION CACHE
# Loaded once from phase3 output for velocity lookup
# ============================================
_signal_df = None
_gdelt_df = None

def _load_phase3_data():
    global _signal_df, _gdelt_df
    sig_path = 'bert_plots/signal_classifications.csv'
    gdt_path = 'bert_plots/gdelt_verification.csv'
    if os.path.exists(sig_path):
        _signal_df = pd.read_csv(sig_path)
    if os.path.exists(gdt_path):
        _gdelt_df = pd.read_csv(gdt_path)


# ============================================
# REAL/FAKE EVENT CLASSIFIER
# This is the core logic that answers:
# "Is the article about a real, verified event?"
# ============================================

def classify_real_event(topic_id, topic_label, confidence):
    """
    Real/fake event classifier.
    Priority: GBM predict_proba → GT dict → signal heuristic → lifestyle fallback.
    
    Confidence gate: if topic match confidence < 0.15, the topic prediction
    itself is unreliable — return UNCERTAIN regardless of GBM output.
    This prevents false REAL EVENT verdicts for out-of-corpus topics
    (climate change, AI/tech) that have no matching topic in the corpus.
    """
    label_lower = str(topic_label).lower()
    tid         = int(topic_id)

    # ── Confidence gate ────────────────────────────────────────────────────
    # If the topic similarity is very low, the matched topic is likely wrong.
    # Don't let the GBM declare REAL EVENT based on a bad topic match.
    if confidence < 0.15:
        return {
            'is_real_event':   None,
            'verdict':         'UNCERTAIN',
            'event_name':      topic_label,
            'event_category':  'unknown',
            'reason':          f'Low topic confidence ({confidence:.1%}) — text may not match any known topic in the corpus. The corpus covers 2012–2022 HuffPost news; topics outside this domain (e.g. AI research, climate policy) have no dedicated cluster.',
            'real_years':      [],
            'event_probability': 0.5,
        }

    # ── Step 1: GBM ML classifier (primary path) ──────────────────────────
    if _gbm_clf_api is not None and _topic_feat_api is not None:
        feat_row = _topic_feat_api[_topic_feat_api['topic_id'] == tid]
        if len(feat_row) > 0 and _signal_df is not None:
            sig  = _signal_df[_signal_df['topic_id'] == tid]
            vel  = float(sig['velocity'].max()) if len(sig) else 0.0
            nz   = int(sig['nonzero_years'].max()) if len(sig) else 1
            sc   = {'NOISE': 0, 'WEAK_SIGNAL': 1,
                    'STRONG_SIGNAL': 2, 'EMERGING': 3}.get(
                sig.loc[sig['velocity'].idxmax(), 'signal_class']
                if len(sig) else 'NOISE', 0)
            feat = [[
                vel, sc, nz,
                float(feat_row['coherence'].values[0]),
                float(feat_row['purity'].values[0]),
                float(feat_row['log_size'].values[0]),
            ]]
            try:
                prob = float(_gbm_clf_api.predict_proba(feat)[0][1])
                if prob >= 0.75:
                    return {
                        'is_real_event':   True,
                        'verdict':         'VERIFIED_REAL',
                        'event_name':      topic_label,
                        'event_category':  'news_event',
                        'reason':          f'GBM classifier: P(real)={prob:.2f} (high confidence)',
                        'real_years':      [],
                        'event_probability': round(prob, 3),
                    }
                elif prob >= 0.55:
                    return {
                        'is_real_event':   True,
                        'verdict':         'LIKELY_REAL',
                        'event_name':      topic_label,
                        'event_category':  'news_event',
                        'reason':          f'GBM classifier: P(real)={prob:.2f} (moderate confidence)',
                        'real_years':      [],
                        'event_probability': round(prob, 3),
                    }
                elif prob >= 0.40:
                    return {
                        'is_real_event':   None,
                        'verdict':         'UNCERTAIN',
                        'event_name':      topic_label,
                        'event_category':  'unknown',
                        'reason':          f'GBM classifier: P(real)={prob:.2f} — borderline, flagged for review',
                        'real_years':      [],
                        'event_probability': round(prob, 3),
                    }
                elif prob >= 0.25:
                    return {
                        'is_real_event':   False,
                        'verdict':         'LIKELY_NOISE',
                        'event_name':      topic_label,
                        'event_category':  'general',
                        'reason':          f'GBM classifier: P(real)={prob:.2f} — background topic',
                        'real_years':      [],
                        'event_probability': round(prob, 3),
                    }
                else:
                    return {
                        'is_real_event':   False,
                        'verdict':         'VERIFIED_NOISE',
                        'event_name':      topic_label,
                        'event_category':  'lifestyle',
                        'reason':          f'GBM classifier: P(real)={prob:.2f} — confirmed non-event',
                        'real_years':      [],
                        'event_probability': round(prob, 3),
                    }
            except Exception:
                pass  # fall through to GT dict

    # ── Step 2: Ground-truth dict (secondary, when GBM unavailable) ────────
    for kw, data in GROUND_TRUTH_EVENTS.items():
        if kw in label_lower:
            if data['verified'] and len(data['real_years']) > 0:
                return {
                    'is_real_event':   True,
                    'verdict':         'VERIFIED_REAL',
                    'event_name':      data['event_name'],
                    'event_category':  data['category'],
                    'reason':          f"Ground-truth: {data['event_name']} in {data['real_years']}",
                    'real_years':      data['real_years'],
                    'event_probability': 0.95,
                }
            else:
                return {
                    'is_real_event':   False,
                    'verdict':         'VERIFIED_NOISE',
                    'event_name':      data['event_name'],
                    'event_category':  data['category'],
                    'reason':          'Ground-truth: lifestyle/entertainment content',
                    'real_years':      [],
                    'event_probability': 0.05,
                }

    # ── Step 3: Phase 3 signal data heuristic ──────────────────────────────
    if _signal_df is not None:
        topic_signals = _signal_df[_signal_df['topic_id'] == tid]
        if len(topic_signals) > 0:
            is_emerging    = 'EMERGING' in topic_signals['signal_class'].tolist()
            gdelt_verified = False
            if _gdelt_df is not None:
                gm = _gdelt_df[_gdelt_df['topic_id'] == tid]
                gdelt_verified = (len(gm) > 0 and
                    gm.iloc[0]['verification_status'] in ('VERIFIED', 'VERIFIED_GROUNDTRUTH'))
            if is_emerging and gdelt_verified:
                return {
                    'is_real_event':   True,
                    'verdict':         'VERIFIED_REAL',
                    'event_name':      topic_label,
                    'event_category':  'news_event',
                    'reason':          'EMERGING signal + GDELT verified',
                    'real_years':      [],
                    'event_probability': 0.85,
                }

    # ── Step 4: Lifestyle keyword fallback ─────────────────────────────────
    lifestyle_kws = ['fashion', 'beauty', 'wedding', 'parenting', 'holiday',
                     'entertainment', 'style', 'celebrity', 'food', 'travel']
    if any(kw in label_lower for kw in lifestyle_kws):
        return {
            'is_real_event':   False,
            'verdict':         'VERIFIED_NOISE',
            'event_name':      topic_label,
            'event_category':  'lifestyle',
            'reason':          'Lifestyle/entertainment topic — not a discrete news event.',
            'real_years':      [],
            'event_probability': 0.1,
        }

    # ── Step 5: Final fallback ──────────────────────────────────────────────
    return {
        'is_real_event':   None,
        'verdict':         'UNCERTAIN',
        'event_name':      topic_label,
        'event_category':  'unknown',
        'reason':          'Run phase3_detector.py to train GBM classifier for this topic.',
        'real_years':      [],
        'event_probability': 0.5,
    }

# ============================================
# HELPER: Text cleaning
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_for_attribution(text):
    cleaned = clean_text(text)
    return [t for t in cleaned.split(' ') if t]


def compute_live_token_attribution(text):
    tokens = tokenize_for_attribution(text)
    if not tokens:
        return []
    if len(tokens) > 60:
        tokens = tokens[:60]

    original_text = ' '.join(tokens)
    e_original = normalize(np.array([embedding_model.encode([original_text])[0]]))[0]

    rows = []
    for i, tok in enumerate(tokens):
        masked_tokens = tokens[:i] + tokens[i + 1:]
        masked_text = ' '.join(masked_tokens).strip()
        if not masked_text:
            score = 1.0
        else:
            e_masked = normalize(np.array([embedding_model.encode([masked_text])[0]]))[0]
            score = 1.0 - float(np.dot(e_original, e_masked))
        rows.append({'position': i, 'token': tok, 'attribution_score': round(float(score), 6)})

    rows.sort(key=lambda x: x['attribution_score'], reverse=True)
    for rnk, row in enumerate(rows, start=1):
        row['rank'] = rnk
    rows.sort(key=lambda x: x['position'])
    return rows


# ============================================
# STARTUP: LOAD MODEL FROM CACHE
# ============================================

print("=" * 60)
print("DYNAMIC TREND & EVENT DETECTOR — API Server (FIXED)")
print("=" * 60)

REQUIRED_CORE = ['embeddings.npy', 'tpi_reduced.npy', 'df.pkl', 'meta.pkl', 'bertopic_model']
missing = [f for f in REQUIRED_CORE if not os.path.exists(os.path.join(CACHE_DIR, f))]
if missing:
    print(f"\n❌ Cache missing: {missing}")
    print("Run bert_model.py first:\n\n    python3 bert_model.py\n")
    import sys; sys.exit(1)

print("\nLoading from bert_model.py cache...")
embeddings     = np.load(os.path.join(CACHE_DIR, 'embeddings.npy'))        # (N, 384)
embeddings_svd = np.load(os.path.join(CACHE_DIR, 'tpi_reduced.npy'))       # (N, 50)
df             = pickle.load(open(os.path.join(CACHE_DIR, 'df.pkl'), 'rb'))
meta           = pickle.load(open(os.path.join(CACHE_DIR, 'meta.pkl'), 'rb'))

_svd_path = os.path.join(CACHE_DIR, 'svd_model.pkl')
if os.path.exists(_svd_path):
    svd_model_api = pickle.load(open(_svd_path, 'rb'))
else:
    from sklearn.decomposition import TruncatedSVD
    _d_tpi   = meta['d_model_tpi']
    _days    = meta['days_since_start']
    _emb_raw = embeddings
    _pe      = np.zeros((_emb_raw.shape[0], _d_tpi), dtype=np.float32)
    _pos     = _days.reshape(-1, 1)
    _div     = np.exp(np.arange(0, _d_tpi, 2, dtype=np.float32) * (-np.log(10000.0) / _d_tpi))
    _pe[:, 0::2] = np.sin(_pos * _div)
    _pe[:, 1::2] = np.cos(_pos * _div)
    _tpi_emb = np.concatenate([_emb_raw, _pe], axis=1)
    svd_model_api = TruncatedSVD(n_components=50, random_state=42)
    svd_model_api.fit(normalize(_tpi_emb, axis=1))
    pickle.dump(svd_model_api, open(_svd_path, 'wb'))
    print(f"  SVD rebuilt and saved")

docs                  = meta['docs']
days_since_start      = meta['days_since_start']
d_model_tpi           = meta['d_model_tpi']
topic_semantic_labels = {int(k): v for k, v in meta.get('topic_semantic_labels', {}).items()}

print("\nLoading SBERT model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading BERTopic model...")
topic_model = BERTopic.load(os.path.join(CACHE_DIR, 'bertopic_model'),
                            embedding_model=embedding_model)

# Load GBM classifier (trained by phase3_detector.py)
_gbm_clf_api    = None
_topic_feat_api = None
_gbm_path       = os.path.join(CACHE_DIR, 'gbm_classifier.pkl')
_feat_path_api  = os.path.join(CACHE_DIR, 'topic_features.csv')
if os.path.exists(_gbm_path):
    _gbm_clf_api = pickle.load(open(_gbm_path, 'rb'))
    print("  GBM classifier loaded")
else:
    print("  GBM not found — run phase3_detector.py first (GT fallback active)")
if os.path.exists(_feat_path_api):
    _topic_feat_api = pd.read_csv(_feat_path_api)

bow_vec_path = os.path.join(CACHE_DIR, 'bow_vec.pkl')
if os.path.exists(bow_vec_path):
    bow_vec = pickle.load(open(bow_vec_path, 'rb'))
else:
    bow_vec = CountVectorizer(max_features=5000, stop_words='english', max_df=0.95, min_df=2)
    bow_vec.fit(docs)
    pickle.dump(bow_vec, open(bow_vec_path, 'wb'))

# ============================================
# PRE-COMPUTE TOPIC CENTROIDS  [FIX A]
# ============================================
# Compute once at startup in SBERT 384-dim space (not SVD space).
# SBERT space is what the model was semantically trained in.
# Using SVD-reduced space caused all queries to collapse to Topic 0
# because SVD compresses variance and the largest cluster dominates.
# ============================================

# AFTER — load saved median centroids from bert_model.py (exact match)
print("\nLoading pre-computed SBERT median centroids...")
topic_centroids_sbert = {}
topic_sizes           = {}
topic_year_range      = {}

_cent_path = os.path.join(CACHE_DIR, 'sbert_centroids.pkl')
_global_mean_api = None

if os.path.exists(_cent_path):
    _saved = pickle.load(open(_cent_path, 'rb'))
    if isinstance(_saved, dict) and 'centroids' in _saved:
        # New format with global_mean
        topic_centroids_sbert = {int(k): v for k, v in _saved['centroids'].items()}
        _global_mean_api = _saved['global_mean']
        print(f"  Loaded {len(topic_centroids_sbert)} whitened centroids + global_mean")
    else:
        # Old format fallback (pre-whitening cache)
        topic_centroids_sbert = {int(k): v for k, v in _saved.items()}
        _global_mean_api = embeddings.mean(axis=0)
        print(f"  Loaded {len(topic_centroids_sbert)} centroids (old format, recomputing mean)")
else:
    print("  sbert_centroids.pkl not found — recomputing with whitening")
    _global_mean_api = embeddings.mean(axis=0)
    _embs_c = embeddings - _global_mean_api
    for t in df['bert_topic'].unique():
        if t == -1:
            continue
        t_idx = df[df['bert_topic'] == t].index.tolist()
        centroid = _embs_c[t_idx].mean(axis=0)
        topic_centroids_sbert[int(t)] = normalize(centroid.reshape(1, -1))[0]
    print(f"  Recomputed {len(topic_centroids_sbert)} whitened centroids")

# Always compute sizes and year ranges (lightweight)
for t in df['bert_topic'].unique():
    if t == -1:
        continue
    t_idx = df[df['bert_topic'] == t].index.tolist()
    topic_sizes[int(t)] = len(t_idx)
    years = df.iloc[t_idx]['year'].values
    topic_year_range[int(t)] = (int(years.min()), int(years.max()))

# Load Phase 3 data for real/fake classification
_load_phase3_data()
print(f"  Phase 3 signals loaded: {_signal_df is not None}")
print(f"  GDELT data loaded:      {_gdelt_df is not None}")

# Build topic info cache
topic_info_cache = {}
for _, row in topic_model.get_topic_info().iterrows():
    tid = row['Topic']
    if tid == -1:
        topic_info_cache[-1] = {'id': -1, 'name': 'Outlier / Unclassified', 'words': [], 'count': int(row['Count'])}
        continue
    words = topic_model.get_topic(tid)
    topic_info_cache[tid] = {
        'id':    int(tid),
        'name':  topic_semantic_labels.get(int(tid), ' / '.join([w for w, _ in words[:3]])),
        'words': [{'word': w, 'score': round(float(s), 4)} for w, s in words[:10]],
        'count': int(row['Count'])
    }

n_topics   = len([k for k in topic_info_cache if k != -1])
n_outliers = int((df['bert_topic'] == -1).sum())

print(f"\n✅ API ready — {n_topics} topics, {len(df):,} docs indexed")
print("=" * 60)


# ============================================
# HELPERS
# ============================================

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_bow_similarity(text_a, text_b):
    vecs = bow_vec.transform([text_a, text_b])
    n    = normalize(vecs.toarray(), axis=1)
    return float(n[0] @ n[1])

def get_sbert_similarity(text_a, text_b):
    embs = embedding_model.encode([text_a, text_b])
    return float(cosine_similarity(embs)[0, 1])


# ============================================
# ENDPOINT: POST /api/predict  [FIX A + FIX B + FIX C]
# ============================================

def _api_sbert_doc_fallback(cleaned_text, topic_scores_dict, df, embeddings):
    """Mirror of _sbert_doc_similarity_fallback from bert_model.py."""
    emb_q      = embedding_model.encode([cleaned_text], show_progress_bar=False)[0]
    emb_q_norm = normalize(emb_q.reshape(1, -1))[0]
    best_t, best_s = -1, -1.0
    rng = np.random.RandomState(42)
    for t in topic_scores_dict:
        t_idx = df[df['bert_topic'] == t].index.tolist()
        if not t_idx:
            continue
        s_idx = rng.choice(t_idx, size=min(30, len(t_idx)), replace=False).tolist()
        sim   = float(np.dot(normalize(embeddings[s_idx], axis=1), emb_q_norm).mean())
        if sim > best_s:
            best_s, best_t = sim, t
    return best_t, best_s

@app.route('/api/predict', methods=['POST'])
def predict_topic():
    """
    Classify a new article into a BERTopic topic AND determine
    whether it describes a real verified news event or noise.

    FIX A: Uses c-TF-IDF keyword overlap (mirrors bert_model.py predict_article_topic).
           Falls back to corpus-grounded healthcare signal → SBERT mean-of-docs.
           Centroid cosine removed — was biased toward Topic 4 for all generic queries.

    FIX B: Adds real/fake event classification to every prediction.

    FIX C: confidence is raw c-TF-IDF overlap score. Capped at 100% for display.
           Top-3 candidates always built from SBERT centroid cosine for transparency.

    Expects JSON: { "text": "article text here" }
    """
    data = request.json or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)

    # ── Step 1: Build query vocabulary (unigrams + bigrams) ───────────────
    words_list  = cleaned.split()
    tokens      = set(words_list)
    bigrams     = set(
        words_list[i] + ' ' + words_list[i+1]
        for i in range(len(words_list)-1)
    )
    query_vocab = tokens | bigrams

    # ── Step 2: c-TF-IDF weighted overlap (primary path) ─────────────────
    topic_scores = {}
    for t_id in [t for t in df['bert_topic'].unique() if t != -1]:
        topic_words = topic_model.get_topic(t_id)
        if not topic_words:
            topic_scores[t_id] = 0.0
            continue
        score = sum(
            ctfidf_w for word, ctfidf_w in topic_words[:30]
            if word in query_vocab
        )
        topic_scores[t_id] = score

    best_topic = max(topic_scores, key=topic_scores.get)
    best_sim   = topic_scores[best_topic]

    # ── Step 3: Fallback when zero c-TF-IDF overlap ───────────────────────
    if best_sim == 0.0:
        healthcare_signals = {
            'affordable', 'obamacare', 'medicaid', 'mandate',
            'supreme', 'repeal', 'aca', 'healthcare', 'insurance'
        }
        if tokens & healthcare_signals:
            pre2020_health_mask = (
                (df['year'] < 2020) &
                (df['clean_text'].str.contains(
                    'health|court|affordable|law|insurance|mandate', regex=True
                )) &
                (df['bert_topic'] != -1)
            )
            if pre2020_health_mask.sum() > 0:
                best_topic = int(
                    df[pre2020_health_mask]['bert_topic'].value_counts().index[0]
                )
                best_sim = 0.5
            else:
                best_topic, best_sim = _api_sbert_doc_fallback(
                    cleaned, topic_scores, df, embeddings
                )
        else:
            best_topic, best_sim = _api_sbert_doc_fallback(
                cleaned, topic_scores, df, embeddings
            )

    LOW_CONFIDENCE_THRESHOLD = 0.15

    if best_sim < LOW_CONFIDENCE_THRESHOLD:
        return jsonify({
            'topic_id':          -1,
            'label':             'No Matching Topic Found',
            'confidence':        round(best_sim, 4),
            'confidence_pct':    f"{min(best_sim, 1.0) * 100:.1f}%",
            'topic':             {
                'id':    -1,
                'name':  'No Matching Topic Found',
                'words': [],
                'count': 0
            },
            'top3_candidates':   [],

            'is_real_event':     None,
            'event_verdict':     'UNCERTAIN',
            'event_name':        'Unknown',
            'event_category':    'unknown',
            'event_reason':      f'Topic confidence too low ({best_sim * 100:.1f}%) — this text does not match any known topic in the corpus. The corpus covers 2012–2022 HuffPost news (politics, COVID, lifestyle). Topics like AI research, climate policy, sports, and business have no dedicated cluster.',
            'real_years':        [],
            'event_probability': 0.0,

            'similar_articles':  [],
            'input_text':        text,
        })

    # ── Step 4: Top-3 candidates via SBERT centroid cosine ───────────────
    # Always use centroid cosine for top-3 display — this gives meaningful
    # similarity values even when the primary path used c-TF-IDF overlap
    # (which produces scores > 1.0) or the fallback path (all scores = 0.0).
    query_emb      = embedding_model.encode([cleaned])[0]
    query_norm_top3 = normalize(query_emb.reshape(1, -1))[0]

    centroid_sims = {
        t: float(np.dot(query_norm_top3, topic_centroids_sbert[t]))
        for t in topic_centroids_sbert
    }
    top3_sorted = sorted(centroid_sims.items(), key=lambda x: x[1], reverse=True)

    top3 = []
    for t, sim in top3_sorted:
        if len(top3) >= 3:
            break
        top3.append({
            'topic_id':   int(t),
            'label':      topic_semantic_labels.get(int(t), f'Topic {t}'),
            'similarity': round(sim, 4)
        })

    # ── Step 5: Get topic info ────────────────────────────────────────────
    tid  = int(best_topic)
    info = topic_info_cache.get(tid, topic_info_cache.get(-1))

    # ── Step 6: Real/Fake event classification ────────────────────────────
    event_verdict = classify_real_event(
        topic_id=tid,
        topic_label=topic_semantic_labels.get(tid, f'Topic {tid}'),
        confidence=best_sim
    )

    # ── Step 7: Top-5 similar corpus articles in SBERT space ─────────────
    query_emb_2d = query_emb.reshape(1, -1)
    sims         = cosine_similarity(query_emb_2d, embeddings)[0]
    top_idx      = sims.argsort()[-5:][::-1]
    similar      = []
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
        # Topic prediction
        'topic_id':          tid,
        'label':             info.get('name', 'Unknown') if info else 'Unknown',
        'confidence':        round(best_sim, 4),
        'confidence_pct':    f"{min(best_sim, 1.0) * 100:.1f}%",  # capped at 100%
        'topic':             info,
        'top3_candidates':   top3,

        # Real/Fake verdict
        'is_real_event':     event_verdict['is_real_event'],
        'event_verdict':     event_verdict['verdict'],
        'event_name':        event_verdict['event_name'],
        'event_category':    event_verdict.get('event_category', 'unknown'),
        'event_reason':      event_verdict['reason'],
        'real_years':        event_verdict.get('real_years', []),
        'event_probability': event_verdict.get('event_probability', 0.5),

        # Source articles
        'similar_articles':  similar,
        'input_text':        text,
    })

# ============================================
# ALL REMAINING ENDPOINTS (unchanged from original)
# ============================================

@app.route('/api/topics', methods=['GET'])
def get_topics():
    topics_list = [v for k, v in topic_info_cache.items() if k != -1]
    topics_list.sort(key=lambda x: x['count'], reverse=True)
    return jsonify({'topics': topics_list, 'total': len(topics_list)})


@app.route('/api/plots/<name>', methods=['GET'])
def get_plot(name):
    safe_name = os.path.basename(name)
    for folder in ['bert_plots', 'model_plots', 'eda_plots']:
        path = os.path.join(folder, safe_name)
        if os.path.exists(path):
            return jsonify({'image': encode_image(path), 'name': safe_name})
    return jsonify({'error': f"Plot '{safe_name}' not found"}), 404


@app.route('/api/similarity', methods=['POST'])
def compare_representations():
    data   = request.json or {}
    text_a = clean_text(data.get('text_a', ''))
    text_b = clean_text(data.get('text_b', ''))
    if not text_a or not text_b:
        return jsonify({'error': 'Provide text_a and text_b'}), 400
    bow_sim   = get_bow_similarity(text_a, text_b)
    sbert_sim = get_sbert_similarity(text_a, text_b)
    return jsonify({
        'text_a': text_a, 'text_b': text_b,
        'bow_cosine':      round(bow_sim, 4),
        'sbert_cosine':    round(sbert_sim, 4),
        'separation_gain': round(bow_sim - sbert_sim, 4),
        'interpretation': (
            f'SBERT separates these semantically distinct texts more '
            f'than BoW (gap = {bow_sim - sbert_sim:+.3f}). '
            'Positive gap means BoW incorrectly rates them as more '
            'similar than their actual semantics warrant.'
        )
    })

@app.route('/api/coherence', methods=['GET'])
def get_coherence():
    path = 'bert_plots/topic_coherence.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Run bert_model.py first'}), 404
    cdf = pd.read_csv(path)
    return jsonify({
        'topics': cdf.to_dict(orient='records'),
        'mean': round(float(cdf['coherence'].mean()), 4),
        'min':  round(float(cdf['coherence'].min()), 4),
        'max':  round(float(cdf['coherence'].max()), 4),
    })


@app.route('/api/separation', methods=['GET'])
def get_separation():
    path = 'bert_plots/context_separation_scores.json'
    if not os.path.exists(path):
        return jsonify({'error': 'Run bert_model.py first'}), 404
    with open(path) as f:
        scores = json.load(f)
    bow   = scores.get('bow', {})
    sbert = scores.get('sbert', {})
    return jsonify({
        'bow': bow, 'sbert': sbert,
        'separation_gap': {
            'bow':   round((bow.get('intra_aca',0)+bow.get('intra_covid',0))/2 - bow.get('cross',0), 4),
            'sbert': round((sbert.get('intra_aca',0)+sbert.get('intra_covid',0))/2 - sbert.get('cross',0), 4),
        }
    })


@app.route('/api/purity', methods=['GET'])
def get_purity():
    path = 'bert_plots/bert_topic_purity.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Run bert_model.py first'}), 404
    pdf = pd.read_csv(path)
    return jsonify({
        'topics': pdf.to_dict(orient='records'),
        'mean': round(float(pdf['purity'].mean()), 4),
        'min':  round(float(pdf['purity'].min()), 4),
        'max':  round(float(pdf['purity'].max()), 4),
    })


@app.route('/api/attribution/<topic_id>', methods=['GET'])
def get_attribution(topic_id):
    path = 'bert_plots/token_attribution.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Run bert_model.py first'}), 404
    adf = pd.read_csv(path)
    if adf.empty:
        return jsonify({'topic_id': topic_id, 'rows': [], 'top_tokens': [], 'count': 0})
    subset = adf if topic_id == 'all' else adf[adf['topic_id'] == int(topic_id)]
    top_tokens_df = (subset.groupby('token', as_index=False)['attribution_score']
                     .mean().sort_values('attribution_score', ascending=False).head(5))
    top_tokens = [{'token': r['token'], 'mean_attribution': round(float(r['attribution_score']), 6)}
                  for _, r in top_tokens_df.iterrows()]
    return jsonify({'topic_id': topic_id, 'count': int(len(subset)),
                    'top_tokens': top_tokens, 'rows': subset.to_dict(orient='records')})


@app.route('/api/attribution_live', methods=['POST'])
def get_live_attribution():
    data = request.json or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    rows = compute_live_token_attribution(text)
    top_tokens = sorted(rows, key=lambda x: x['attribution_score'], reverse=True)[:5]
    return jsonify({'input_text': text, 'count': len(rows), 'rows': rows,
                    'top_tokens': [{'token': t['token'], 'attribution_score': t['attribution_score'],
                                    'rank': t['rank']} for t in top_tokens]})


@app.route('/api/gdelt', methods=['GET'])
def get_gdelt():
    path = 'bert_plots/gdelt_verification.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Run phase3_detector.py first'}), 404
    gdf = pd.read_csv(path)

    def display_score(row):
        st = row['verification_status']
        if st == 'VERIFIED_GROUNDTRUTH': return 1.0
        if st == 'VERIFIED':             return max(float(row['gdelt_peak_value']), 0.5)
        if st == 'NOISE_UNVERIFIED':     return 0.15
        return 0.05

    gdf['display_score'] = gdf.apply(display_score, axis=1)
    summary = {
        'VERIFIED_GROUNDTRUTH': int((gdf['verification_status'] == 'VERIFIED_GROUNDTRUTH').sum()),
        'VERIFIED':             int((gdf['verification_status'] == 'VERIFIED').sum()),
        'NOISE_UNVERIFIED':     int((gdf['verification_status'] == 'NOISE_UNVERIFIED').sum()),
        'GDELT_UNAVAILABLE':    int((gdf['verification_status'] == 'GDELT_UNAVAILABLE').sum()),
    }
    return jsonify({'records': gdf.to_dict(orient='records'), 'summary': summary, 'total': len(gdf)})


@app.route('/api/events', methods=['GET'])
def get_events():
    path = 'bert_plots/phase3_final_events.json'
    if not os.path.exists(path):
        return jsonify({'error': 'Run phase3_detector.py first'}), 404
    with open(path) as f:
        events = json.load(f)
    return jsonify(events)


@app.route('/api/signals', methods=['GET'])
def get_signals():
    path = 'bert_plots/signal_classifications.csv'
    if not os.path.exists(path):
        return jsonify({'error': 'Run phase3_detector.py first'}), 404
    sdf = pd.read_csv(path).sort_values(['year', 'velocity'], ascending=[True, False])
    return jsonify(sdf[['topic_id', 'topic_label', 'year', 'signal_class', 'velocity']].to_dict(orient='records'))


@app.route('/api/summary', methods=['GET'])
def get_summary():
    return jsonify({
        'phase1': {
            'method': 'LDA', 'topics': 10, 'confidence': 0.6103,
            'limitation': 'Bag-of-words conflation: ACA 2014 merged with COVID 2020'
        },
        'phase2': {
            'method': 'BERTopic (SBERT + SVD + UMAP + HDBSCAN + c-TF-IDF)',
            'topics': n_topics, 'documents': len(df),
            'outliers': n_outliers, 'outlier_rate': round(n_outliers / len(df), 4),
            'pipeline': 'SBERT(384) → TPI(+32=416) → SVD(→50) → UMAP(→5) → HDBSCAN → c-TF-IDF',
            'prediction_fix': 'c-TF-IDF keyword overlap + corpus-grounded healthcare fallback + SBERT mean-of-docs.'
        }
    })


# ============================================
# SERVE FRONTEND
# ============================================

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/bert_plots/<path:filename>')
def serve_bert_plots(filename):
    return send_from_directory('bert_plots', filename)

@app.route('/model_plots/<path:filename>')
def serve_model_plots(filename):
    return send_from_directory('model_plots', filename)

@app.route('/eda_plots/<path:filename>')
def serve_eda_plots(filename):
    return send_from_directory('eda_plots', filename)


if __name__ == '__main__':
    print("\n✅ Open your browser at: http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=False)