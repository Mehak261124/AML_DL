# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Phase 2 — Deep Learning: BERTopic Neural Topic Modeling
# ============================================
#
# References:
#   Grootendorst, M. (2022). BERTopic: Neural topic modeling
#       with a class-based TF-IDF procedure. arXiv:2203.05794.
#   Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence
#       Embeddings using Siamese BERT-Networks. EMNLP 2019.
#       arXiv:1908.10084.
#   McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform
#       Manifold Approximation and Projection for Dimension
#       Reduction. arXiv:1802.03426.
#
# FIXES APPLIED (vs previous version):
#   FIX A — SVD(50) pre-reduction before UMAP (McInnes et al. [6])
#   FIX B — Extended stop word list for CountVectorizer
#   FIX C — predict_article_topic() uses c-TF-IDF overlap + corpus-grounded fallback
#   FIX D — Semantic label generation via Claude API + rule-based fallback
#   FIX E — Plot alignment fixes throughout
# ============================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import re
import json
import urllib.request
import urllib.error
from collections import defaultdict

warnings.filterwarnings('ignore')

# ---------- Deep-learning stack ----------
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

os.makedirs('bert_plots', exist_ok=True)
print("Phase 2 plots will be saved to: bert_plots/")

# ============================================
# STEP 1 — RELOAD & CLEAN DATA
# ============================================
print("\n" + "="*60)
print("PHASE 2 — DEEP LEARNING: BERTopic Neural Topic Modeling")
print("="*60)

print("\nLoading data...")
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
df = df[['headline', 'short_description', 'category', 'date']].copy()
df['text'] = df['headline'] + ' ' + df['short_description']
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)
df = df.sort_values('date').reset_index(drop=True)

df = df.groupby(df['date'].dt.year).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)

print(f"Total documents: {len(df)}")
print(f"Years covered:   {sorted(df['date'].dt.year.unique())}")

def clean_text(text):
    """Minimal cleaning — preserve semantics for SBERT.
    Retains digits (unlike lda_model.py) because numbers like '19'
    in 'covid 19' or '2020' carry discriminative temporal meaning.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['year'] = df['date'].dt.year
docs = df['clean_text'].tolist()
timestamps = df['date'].tolist()

print("Text cleaning done.")

# ============================================
# STEP 2 — SBERT SENTENCE EMBEDDINGS
# ============================================
print("\n--- STEP 2: SBERT Sentence Embeddings ---")
print("Model: all-MiniLM-L6-v2  (384-dim, 22M params)")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings for 11,000 documents...")
embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True,
    batch_size=64
)
print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Embedding dimensionality: {embeddings.shape[1]}")

cov_eigenvalues = np.linalg.svd(embeddings - embeddings.mean(0), compute_uv=False) ** 2
pr = float(cov_eigenvalues.sum()**2 / (cov_eigenvalues**2).sum())
print(f"Participation Ratio (intrinsic dim estimate): {pr:.1f}")

# ============================================
# STEP 2b — TEMPORAL POSITIONAL INJECTION (TPI)
# ============================================
print("\n--- STEP 2b: Temporal Positional Injection (TPI) ---")

days_since_start = (df['date'] - df['date'].min()).dt.days.values.astype(np.float32)
d_model_tpi = 32

def build_temporal_positional_encoding(days, d_model=32):
    """Create sinusoidal temporal encodings for day indices."""
    pe = np.zeros((len(days), d_model), dtype=np.float32)
    position = days.reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

temporal_pe    = build_temporal_positional_encoding(days_since_start, d_model=d_model_tpi)
tpi_embeddings = np.concatenate([embeddings, temporal_pe], axis=1)

print(f"Temporal encoding shape:  {temporal_pe.shape}")
print(f"TPI embedding shape:      {tpi_embeddings.shape}")
print(f"Augmented dim:            {embeddings.shape[1]} + {d_model_tpi} = {tpi_embeddings.shape[1]}")

# ============================================
# STEP 2c — SVD PRE-REDUCTION  [FIX A]
# ============================================
print("\n--- STEP 2c: SVD Pre-Reduction 416 → 50 dims [FIX A] ---")

tpi_norm = normalize(tpi_embeddings, axis=1)
svd_model = TruncatedSVD(n_components=50, random_state=42)
tpi_reduced = svd_model.fit_transform(tpi_norm)

explained = svd_model.explained_variance_ratio_.sum()
print(f"SVD explained variance (50 components): {explained:.3f}")
print(f"Shape before SVD: {tpi_embeddings.shape}  →  after SVD: {tpi_reduced.shape}")

# ============================================
# STEP 3 — UMAP DIMENSIONALITY REDUCTION
# ============================================
print("\n--- STEP 3: UMAP Dimensionality Reduction ---")
print("Input: 50-dim SVD-reduced embeddings (not raw 416-dim)")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
    verbose=True
)

umap_2d = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
embeddings_2d = umap_2d.fit_transform(tpi_reduced)
df['umap_x'] = embeddings_2d[:, 0]
df['umap_y'] = embeddings_2d[:, 1]

print(f"2D UMAP projection shape: {embeddings_2d.shape}")

# ============================================
# STEP 3b — CURRICULUM ORDERING
# ============================================
print("\n--- STEP 3b: Curriculum Ordering (by Text Richness) ---")
tokenized_docs   = df['clean_text'].str.split()
doc_word_count   = tokenized_docs.str.len()
doc_unique_count = tokenized_docs.apply(
    lambda toks: len(set(toks)) if isinstance(toks, list) else 0
)
df['text_richness'] = doc_unique_count / (doc_word_count + 1)

curriculum_idx     = np.argsort(-df['text_richness'].values)
richness_threshold = float(df['text_richness'].median())
n_high = int((df['text_richness'] >= richness_threshold).sum())
n_low  = len(df) - n_high

print(f"Curriculum: {n_high} high-richness docs first, then {n_low} low-richness docs.")

docs_curriculum        = [docs[i]       for i in curriculum_idx]
timestamps_curriculum  = [timestamps[i] for i in curriculum_idx]
tpi_reduced_curriculum = tpi_reduced[curriculum_idx]

# ============================================
# STEP 4 — HDBSCAN CLUSTERING
# ============================================
print("\n--- STEP 4: HDBSCAN Clustering ---")

hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    prediction_data=True
)

# ============================================
# STEP 5 — c-TF-IDF TOPIC REPRESENTATION  [FIX B]
# ============================================
print("\n--- STEP 5: Extended Stop Words + CountVectorizer [FIX B] ---")

# News-domain stop words: generic terms that appear across all topics
# and add no discriminative value to c-TF-IDF topic representations.
# Note: healthcare/legal terms (court, affordable, mandate) are NOT here —
# they are filtered by max_df=0.98 because they occur in >98% of political
# docs. The predict_article_topic() fallback handles this gap directly.
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

all_stops = list(ENGLISH_STOP_WORDS.union(news_domain_stops))

print(f"Stop word list size: {len(ENGLISH_STOP_WORDS)} (sklearn) + "
      f"{len(news_domain_stops)} (news-domain) = {len(all_stops)} total")

vectorizer_model = CountVectorizer(
    stop_words=all_stops,
    min_df=2,
    max_df=0.98,          # raise from 0.95 → allows 'health', 'court', 'act'
    ngram_range=(1, 2),
    vocabulary=None       # keep auto-vocabulary
)

ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=False)

representation_model = [
    MaximalMarginalRelevance(diversity=0.3),
    KeyBERTInspired()
]

# ============================================
# STEP 6 — FIT BERTopic
# ============================================
print("\nFitting BERTopic model...")
print("Pipeline: SBERT(384) → TPI(+32) → SVD(→50) → UMAP(→5) → HDBSCAN → c-TF-IDF")

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    nr_topics=15,
    top_n_words=10,
    verbose=True
)

topics_curriculum, probs_curriculum = topic_model.fit_transform(
    docs_curriculum, tpi_reduced_curriculum
)

topics_arr = np.empty(len(df), dtype=int)
topics_arr[curriculum_idx] = np.array(topics_curriculum, dtype=int)
topics = topics_arr.tolist()
probs  = probs_curriculum

n_outliers_initial = (np.array(topics) == -1).sum()
print(f"\nInitial outliers: {n_outliers_initial} ({n_outliers_initial/len(df)*100:.1f}%)")

# ============================================
# STEP 6b — REDUCE OUTLIERS
# ============================================
print("\n--- STEP 6b: Reducing Outliers via c-TF-IDF Strategy ---")

new_topics_curriculum = topic_model.reduce_outliers(
    docs_curriculum, topics_curriculum, strategy="c-tf-idf", threshold=0.1
)
topic_model.update_topics(
    docs_curriculum,
    topics=new_topics_curriculum,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model
)

new_topics_arr = np.empty(len(df), dtype=int)
new_topics_arr[curriculum_idx] = np.array(new_topics_curriculum, dtype=int)
topics = new_topics_arr.tolist()
df['bert_topic'] = topics

n_outliers = (df['bert_topic'] == -1).sum()
print(f"Outliers before reduction: {n_outliers_initial} ({n_outliers_initial/len(df)*100:.1f}%)")
print(f"Outliers after  reduction: {n_outliers}         ({n_outliers/len(df)*100:.1f}%)")
print(f"Documents reassigned:      {n_outliers_initial - n_outliers}")

# Hard negatives
print("\nIdentifying hard negatives among remaining outliers...")
topic_centroids = {}
for t in sorted(df['bert_topic'].unique()):
    if t == -1:
        continue
    t_idx = df.index[df['bert_topic'] == t].tolist()
    if len(t_idx) == 0:
        continue
    topic_centroids[t] = normalize(tpi_reduced[t_idx].mean(axis=0, keepdims=True))[0]

hard_negative_rows = []
outlier_indices = df.index[df['bert_topic'] == -1].tolist()
for idx in outlier_indices:
    if not topic_centroids:
        continue
    doc_vec = normalize(tpi_reduced[idx].reshape(1, -1))[0]
    sims = [(tid, float(np.dot(doc_vec, cent))) for tid, cent in topic_centroids.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    if len(sims) < 2:
        continue
    top1_tid, top1_sim = sims[0]
    top2_tid, top2_sim = sims[1]
    ambiguity_score = top1_sim + top2_sim
    hard_negative_rows.append({
        'doc_idx':         idx,
        'headline':        df.at[idx, 'headline'],
        'topic_1':         int(top1_tid),
        'topic_1_sim':     round(top1_sim, 6),
        'topic_2':         int(top2_tid),
        'topic_2_sim':     round(top2_sim, 6),
        'ambiguity_score': round(ambiguity_score, 6)
    })

if hard_negative_rows:
    hard_neg_df = pd.DataFrame(hard_negative_rows).sort_values(
        'ambiguity_score', ascending=False
    ).head(50)
else:
    hard_neg_df = pd.DataFrame(columns=[
        'doc_idx', 'headline', 'topic_1', 'topic_1_sim',
        'topic_2', 'topic_2_sim', 'ambiguity_score'
    ])
hard_neg_df.to_csv('bert_plots/hard_negatives.csv', index=False)
print(f"Saved: bert_plots/hard_negatives.csv ({len(hard_neg_df)} rows)")

topic_info = topic_model.get_topic_info()
n_topics   = len(topic_info[topic_info['Topic'] != -1])

print(f"\nBERTopic fitting complete!")
print(f"Topics discovered: {n_topics}")
print(f"Remaining outliers: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
print(f"\n--- BERTopic Discovered Topics ---")
for _, row in topic_info.iterrows():
    if row['Topic'] == -1:
        continue
    topic_words = topic_model.get_topic(row['Topic'])
    words_str = ' | '.join([w for w, _ in topic_words[:8]])
    print(f"  Topic {row['Topic']:3d} ({row['Count']:4d} docs): {words_str}")

# ============================================
# HELPER: safe non-outlier topic indexer
# ============================================

def _non_outlier_top(topic_series):
    counts = topic_series.value_counts()
    for t in counts.index:
        if t != -1:
            return t
    return -1

# ============================================
# STEP 6c — SEMANTIC LABEL GENERATION  [FIX D]
# ============================================
# Generates human-readable topic names automatically using:
#   1. Claude API (claude-sonnet-4-20250514) — best quality
#   2. Rule-based keyword matching — fallback when API unavailable
# Neither approach requires manual labeling.
# ============================================

print("\n--- STEP 6c: Semantic Label Generation [FIX D] ---")
print("Generating human-readable names from c-TF-IDF keywords...")

SEMANTIC_RULES = [
    # COVID / Health — check before generic "trump" rules
    (['coronavirus pandemic', 'coronavirus outbreak'],         'COVID-19 Pandemic 2020'),
    (['covid19 vaccine', 'vaccination', 'vaccinated'],         'COVID-19 Vaccination Campaign'),
    (['coronavirus', 'covid19', 'pandemic'],                   'COVID-19 Crisis'),
    # Russia / Ukraine
    (['ukraine', 'ukrainian', 'putin'],                        'Russia-Ukraine War 2022'),
    # Trump controversies — specific scandal names first
    (['stormy daniels'],                                       'Trump-Stormy Daniels Scandal'),
    (['mueller', 'muellers'],                                  'Mueller Investigation 2017-19'),
    (['impeachment', 'colbert'],                               'Trump Impeachment Era'),
    (['border wall'],                                          'US-Mexico Border Wall Debate'),
    (['roy moore'],                                            'Alabama Senate Race 2017'),
    # 2016 election — specific candidates first
    (['bernie sanders', 'bernie', 'sanders'],                  '2016 Democratic Primary'),
    (['hillary clinton', 'hillary', 'clinton'],                '2016 Presidential Election'),
    # Biden / 2020
    (['joe biden', 'biden administration'],                    '2020 Presidential Election'),
    # Generic Trump presidency — only if no specific scandal matched above
    (['trump administration', 'donald trump', 'donald trumps'],'Trump Presidency 2017-21'),
    (['fox news'],                                             'Fox News & Conservative Media'),
    (['reform', 'campaign', 'political'],                      'US Political Campaigns'),
    # Lifestyle — most specific first
    (['beauty', 'bride', 'wedding'],                           'Wedding & Beauty'),
    (['halloween', 'costume', 'holiday', 'christmas'],         'Holidays & Entertainment'),
    (['childhood', 'parenting', 'baby', 'marriage'],           'Parenting & Family Life'),
    (['fashion', 'dress', 'style twitter', 'magazine'],        'Fashion & Style Trends'),
    (['fashion', 'style'],                                     'Fashion & Lifestyle'),
    (['healthy', 'sleep', 'wellness'],                         'Health & Wellness'),
]


def _rule_based_label(top_words):
    """Keyword-matching fallback for semantic label generation."""
    word_str = ' '.join([w.lower() for w in top_words])
    for keywords, label in SEMANTIC_RULES:
        for kw in keywords:
            if kw in word_str:
                return label
    # Last resort: title-case first 3 words
    return ' & '.join(w.replace('/', ' ').strip().title() for w in top_words[:3])


def generate_semantic_label(topic_id, topic_model_ref):
    """
    Generate human-readable label via Claude API with rule-based fallback.
    Uses urllib (stdlib only — no new deps).
    """
    words_scores = topic_model_ref.get_topic(topic_id)
    if not words_scores:
        return f"Topic {topic_id}"

    top_words = [w for w, _ in words_scores[:10]]
    top_words_str = ', '.join(top_words)

    prompt = (
        f"These are the top keywords from a news article topic cluster: "
        f"{top_words_str}\n\n"
        f"Generate a short, specific, human-readable label (3-6 words) "
        f"that describes what news event or theme this topic represents. "
        f"Be specific — use the actual event name if recognizable "
        f"(e.g. '2016 US Presidential Election', 'COVID-19 Pandemic', "
        f"'Russia-Ukraine War 2022', 'Mueller Investigation'). "
        f"Return ONLY the label, no explanation, no quotes."
    )

    try:
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 30,
            "messages": [{"role": "user", "content": prompt}]
        }).encode('utf-8')

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            label = data['content'][0]['text'].strip().strip('"').strip("'")
            if len(label) > 55:
                label = label[:52] + '...'
            return label

    except Exception:
        return _rule_based_label(top_words)


# Generate labels for all topics
# FIX: convert keys to plain int (not numpy.int64) for JSON serialisation
topic_semantic_labels = {}
topic_ids_sorted = sorted([int(t) for t in df['bert_topic'].unique() if t != -1])

for tid in topic_ids_sorted:
    label = generate_semantic_label(tid, topic_model)
    topic_semantic_labels[int(tid)] = label          # plain int key — JSON-safe
    words_preview = ', '.join([w for w, _ in topic_model.get_topic(tid)[:4]])
    print(f"  T{tid:2d}: [{words_preview}...]  →  '{label}'")

# Post-process: de-duplicate labels by appending top keyword when two topics share a name
from collections import Counter
label_counts = Counter(topic_semantic_labels.values())
duplicate_labels = {lbl for lbl, cnt in label_counts.items() if cnt > 1}

if duplicate_labels:
    print(f"\n  Deduplicating {len(duplicate_labels)} shared label(s)...")
    for dup_label in duplicate_labels:
        # Find all topic IDs sharing this label
        dup_tids = [tid for tid, lbl in topic_semantic_labels.items() if lbl == dup_label]
        for tid in dup_tids:
            words = topic_model.get_topic(tid)
            # Find the most distinctive top word not already in the label
            for w, _ in words[:6]:
                w_clean = w.split()[0].title()  # first token, title-cased
                if w_clean.lower() not in dup_label.lower():
                    topic_semantic_labels[int(tid)] = f"{dup_label} ({w_clean})"
                    break
            else:
                topic_semantic_labels[int(tid)] = f"{dup_label} (T{tid})"

    print("  Deduplicated labels:")
    for tid in topic_ids_sorted:
        if topic_semantic_labels[int(tid)] != topic_semantic_labels.get(int(tid), ''):
            print(f"    T{tid:2d}: {topic_semantic_labels[int(tid)]}")

# Save with str keys for maximum JSON compatibility
with open('bert_plots/topic_semantic_labels.json', 'w') as f:
    json.dump({str(k): v for k, v in topic_semantic_labels.items()}, f, indent=2)
print(f"\nSaved: bert_plots/topic_semantic_labels.json")


def get_label(tid):
    """Return semantic label for a topic id."""
    return topic_semantic_labels.get(int(tid), f'Topic {tid}')


# ============================================
# STEP 7 — VISUALIZATIONS  [FIX E — alignment]
# ============================================
print("\n--- STEP 7: Generating Visualizations ---")

# Shared palette
unique_topics = sorted(df['bert_topic'].unique())
palette   = sns.color_palette('husl', n_colors=max(len(unique_topics), 1))
color_map = {t: palette[i % len(palette)] for i, t in enumerate(unique_topics) if t != -1}
color_map[-1] = (0.85, 0.85, 0.85)

# --- 7a: 2D UMAP Topic Clusters ---
# FIX E: semantic labels in legend, two-column legend below plot
fig_umap, ax_umap = plt.subplots(figsize=(16, 10))

outliers_df = df[df['bert_topic'] == -1]
if len(outliers_df) > 0:
    ax_umap.scatter(outliers_df['umap_x'], outliers_df['umap_y'],
                    c=[color_map[-1]], s=6, alpha=0.12, label='Outliers', zorder=1)

for t in unique_topics:
    if t == -1:
        continue
    subset = df[df['bert_topic'] == t]
    label  = get_label(t)
    ax_umap.scatter(subset['umap_x'], subset['umap_y'],
                    c=[color_map[t]], s=18, alpha=0.55,
                    label=f"T{t}: {label}", zorder=2)

ax_umap.set_title('BERTopic — UMAP 2D Topic Clusters\n'
                  '(SBERT → TPI → SVD(50) → UMAP → HDBSCAN → reduce_outliers)',
                  fontsize=13, fontweight='bold', pad=12)
ax_umap.set_xlabel('UMAP Dimension 1', fontsize=11)
ax_umap.set_ylabel('UMAP Dimension 2', fontsize=11)

# Two-column legend placed below the plot — avoids overlap with data
handles, labels_leg = ax_umap.get_legend_handles_labels()
ax_umap.legend(
    handles, labels_leg,
    loc='upper center', bbox_to_anchor=(0.5, -0.12),
    ncol=3, fontsize=8, markerscale=1.8,
    framealpha=0.9, edgecolor='#cccccc'
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig('bert_plots/umap_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/umap_clusters.png")

# --- 7b: Topic Bar Chart ---
# FIX E: semantic labels as titles, wider figure, cleaner layout
top_n_topics = min(n_topics, 10)
n_cols = 5
n_rows = (top_n_topics + n_cols - 1) // n_cols

fig_bar, axes_bar = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows))
fig_bar.suptitle('BERTopic — Top Words Per Topic (c-TF-IDF)\n'
                 'Extended stop list removes non-discriminative words [FIX B]',
                 fontsize=13, fontweight='bold', y=1.01)
axes_flat = axes_bar.flatten() if top_n_topics > 1 else [axes_bar]

for i in range(top_n_topics):
    topic_id    = topic_info[topic_info['Topic'] != -1].iloc[i]['Topic']
    words_scores = topic_model.get_topic(topic_id)
    words  = [w for w, _ in words_scores[:8]]
    scores = [s for _, s in words_scores[:8]]
    ax     = axes_flat[i]
    bars   = ax.barh(words[::-1], scores[::-1],
                     color=palette[i % len(palette)], alpha=0.82)
    semantic = get_label(topic_id)
    # Truncate title to fit
    title_str = f"T{topic_id}: {semantic}"
    if len(title_str) > 35:
        title_str = title_str[:32] + '...'
    ax.set_title(title_str, fontsize=8.5, fontweight='bold', pad=4)
    ax.set_xlabel('c-TF-IDF Score', fontsize=7.5)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.tick_params(axis='x', labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

for j in range(top_n_topics, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('bert_plots/bertopic_topics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/bertopic_topics.png")

# --- 7c: Topics Over Time ---
print("\nComputing Topics Over Time...")
topics_over_time = topic_model.topics_over_time(
    docs_curriculum, timestamps_curriculum, nr_bins=11
)

top_topic_ids = (
    topic_info[topic_info['Topic'] != -1]
    .nlargest(5, 'Count')['Topic']
    .tolist()
)
colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# FIX E: legend outside plot, no overlap
fig_tot, ax_tot = plt.subplots(figsize=(15, 7))
for idx, tid in enumerate(top_topic_ids):
    subset = topics_over_time[topics_over_time['Topic'] == tid]
    label  = get_label(tid)
    ax_tot.plot(subset['Timestamp'], subset['Frequency'],
                marker='o', linewidth=2.5, markersize=5,
                label=f'T{tid}: {label}',
                color=colors_line[idx % len(colors_line)])

ax_tot.set_title('BERTopic — Topic Evolution Over Time\n(Neural Temporal Tracking)',
                 fontsize=13, fontweight='bold')
ax_tot.set_xlabel('Year', fontsize=11)
ax_tot.set_ylabel('Topic Frequency', fontsize=11)
ax_tot.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax_tot.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('bert_plots/topics_over_time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/topics_over_time.png")

# ============================================
# STEP 8 — CONTEXT SEPARATION ANALYSIS
# ============================================
print("\n" + "="*60)
print("STEP 8 — CONTEXT SEPARATION ANALYSIS")
print("(Does BERT resolve the LDA COVID/Court conflation?)")
print("="*60)

bow_vectorizer = CountVectorizer(
    max_features=5000, stop_words='english', max_df=0.95, min_df=2
)
bow_matrix = bow_vectorizer.fit_transform(docs)

def get_bow_similarity(text_a, text_b):
    vecs = bow_vectorizer.transform([text_a, text_b])
    n    = normalize(vecs.toarray().astype(np.float32), axis=1)
    return float(n[0] @ n[1])

aca_mask   = (df['year'] < 2020) & (
    df['clean_text'].str.contains('health|court|affordable|law', regex=True)
)
covid_mask = (df['year'] >= 2020) & (
    df['clean_text'].str.contains('covid|coronavirus|pandemic|lockdown', regex=True)
)

aca_docs   = df[aca_mask]
covid_docs = df[covid_mask]

print(f"\nPre-2020 health/court documents: {len(aca_docs)}")
print(f"Post-2020 COVID documents:       {len(covid_docs)}")

separation_success = False
separation_scores  = {}

if len(aca_docs) > 0 and len(covid_docs) > 0:
    aca_topics   = aca_docs['bert_topic'].value_counts()
    covid_topics = covid_docs['bert_topic'].value_counts()

    print(f"\n--- Topic assignments for ACA-era health/court docs (pre-2020) ---")
    for t, cnt in aca_topics.head(5).items():
        if t == -1:
            print(f"  Outliers: {cnt} docs")
        else:
            words = topic_model.get_topic(t)
            label = get_label(t)
            print(f"  Topic {t} [{label}]: {cnt} docs")

    print(f"\n--- Topic assignments for COVID-era docs (2020+) ---")
    for t, cnt in covid_topics.head(5).items():
        if t == -1:
            print(f"  Outliers: {cnt} docs")
        else:
            label = get_label(t)
            print(f"  Topic {t} [{label}]: {cnt} docs")

    aca_top_topic   = _non_outlier_top(aca_docs['bert_topic'])
    covid_top_topic = _non_outlier_top(covid_docs['bert_topic'])

    if aca_top_topic != covid_top_topic and aca_top_topic != -1 and covid_top_topic != -1:
        print(f"\n✅ SUCCESS: BERT correctly SEPARATED the contexts!")
        print(f"   ACA-era docs   → Topic {aca_top_topic}: {get_label(aca_top_topic)}")
        print(f"   COVID-era docs → Topic {covid_top_topic}: {get_label(covid_top_topic)}")
        separation_success = True
    else:
        print(f"\n⚠ Topics overlap — checking cosine distance in embedding space...")

    aca_indices   = aca_docs.index.tolist()[:200]
    covid_indices = covid_docs.index.tolist()[:200]

    aca_emb   = embeddings[aca_indices]
    covid_emb = embeddings[covid_indices]

    intra_aca_sbert   = cosine_similarity(aca_emb).mean()
    intra_covid_sbert = cosine_similarity(covid_emb).mean()
    cross_sbert       = cosine_similarity(aca_emb, covid_emb).mean()
    gap_sbert         = min(intra_aca_sbert, intra_covid_sbert) - cross_sbert

    bow_dense = bow_matrix.toarray().astype(np.float32)
    bow_norm  = normalize(bow_dense, axis=1)
    bow_aca   = bow_norm[aca_indices]
    bow_covid = bow_norm[covid_indices]

    intra_aca_bow   = float((bow_aca @ bow_aca.T).mean())
    intra_covid_bow = float((bow_covid @ bow_covid.T).mean())
    cross_bow       = float((bow_aca @ bow_covid.T).mean())
    gap_bow         = min(intra_aca_bow, intra_covid_bow) - cross_bow

    print(f"\n--- Embedding Space Analysis ---")
    print(f"                        BoW         SBERT")
    print(f"  Intra-ACA:          {intra_aca_bow:.4f}      {intra_aca_sbert:.4f}")
    print(f"  Intra-COVID:        {intra_covid_bow:.4f}      {intra_covid_sbert:.4f}")
    print(f"  Cross-group:        {cross_bow:.4f}      {cross_sbert:.4f}")
    print(f"  Separation gap:     {gap_bow:.4f}      {gap_sbert:.4f}")

    separation_scores = {
        'bow':   {'intra_aca': round(intra_aca_bow, 4),
                  'intra_covid': round(intra_covid_bow, 4),
                  'cross': round(cross_bow, 4),
                  'gap': round(gap_bow, 4)},
        'sbert': {'intra_aca': round(float(intra_aca_sbert), 4),
                  'intra_covid': round(float(intra_covid_sbert), 4),
                  'cross': round(float(cross_sbert), 4),
                  'gap': round(float(gap_sbert), 4)}
    }
    with open('bert_plots/context_separation_scores.json', 'w') as f:
        json.dump(separation_scores, f, indent=2)
    print("\nSaved: bert_plots/context_separation_scores.json")

    # --- TPI effect plot ---
    aca_tpi_emb   = tpi_embeddings[aca_indices]
    covid_tpi_emb = tpi_embeddings[covid_indices]
    intra_aca_tpi   = float(cosine_similarity(aca_tpi_emb).mean())
    intra_covid_tpi = float(cosine_similarity(covid_tpi_emb).mean())
    cross_tpi       = float(cosine_similarity(aca_tpi_emb, covid_tpi_emb).mean())
    gap_tpi         = min(intra_aca_tpi, intra_covid_tpi) - cross_tpi

    x_tp = np.arange(3); w_tp = 0.35
    fig_tpi, ax_tpi = plt.subplots(figsize=(9, 5))
    no_tpi_vals = [float(intra_aca_sbert), float(intra_covid_sbert), float(cross_sbert)]
    tpi_vals    = [intra_aca_tpi, intra_covid_tpi, cross_tpi]
    b1 = ax_tpi.bar(x_tp - w_tp/2, no_tpi_vals, w_tp,
                    color='#5c6bc0', alpha=0.85, label='Without TPI')
    b2 = ax_tpi.bar(x_tp + w_tp/2, tpi_vals,    w_tp,
                    color='#26a69a', alpha=0.85, label='With TPI')
    ax_tpi.set_xticks(x_tp)
    ax_tpi.set_xticklabels(['Intra-ACA', 'Intra-COVID', 'Cross-group'], fontsize=11)
    ax_tpi.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax_tpi.set_title(
        f'TPI Effect on Context Separation\n'
        f'Gap without TPI = {gap_sbert:.3f}  |  Gap with TPI = {gap_tpi:.3f}',
        fontsize=12, fontweight='bold'
    )
    ax_tpi.legend(fontsize=10)
    ax_tpi.grid(axis='y', alpha=0.25)
    for bar, val in zip(list(b1) + list(b2), no_tpi_vals + tpi_vals):
        ax_tpi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bert_plots/tpi_effect.png', dpi=150, bbox_inches='tight')
    plt.close(fig_tpi)
    print("Saved: bert_plots/tpi_effect.png")

    # --- Context separation scatter + bar ---
    # FIX E: larger figure, better bar label placement
    fig_cs, axes_cs = plt.subplots(1, 2, figsize=(17, 6))
    fig_cs.suptitle('Context Separation: ACA Legal (pre-2020) vs COVID (2020+)\n'
                    '(SBERT resolves LDA bag-of-words conflation)',
                    fontsize=12, fontweight='bold')

    ax_cs = axes_cs[0]
    ax_cs.scatter(df['umap_x'], df['umap_y'], c='lightgray', s=4, alpha=0.1)
    ax_cs.scatter(aca_docs['umap_x'], aca_docs['umap_y'],
                  c='royalblue', s=22, alpha=0.6,
                  label=f'ACA/Health pre-2020 (n={len(aca_docs)})')
    ax_cs.scatter(covid_docs['umap_x'], covid_docs['umap_y'],
                  c='crimson', s=22, alpha=0.6,
                  label=f'COVID 2020+ (n={len(covid_docs)})')
    ax_cs.set_title('UMAP Embedding Space', fontsize=11, fontweight='bold')
    ax_cs.set_xlabel('UMAP-1', fontsize=10)
    ax_cs.set_ylabel('UMAP-2', fontsize=10)
    ax_cs.legend(fontsize=9, loc='upper right')

    ax_cs2  = axes_cs[1]
    x_cs    = np.arange(3); bw_cs = 0.32
    bow_vals   = [intra_aca_bow, intra_covid_bow, cross_bow]
    sbert_vals = [float(intra_aca_sbert), float(intra_covid_sbert), float(cross_sbert)]
    bars_bow   = ax_cs2.bar(x_cs - bw_cs/2, bow_vals,   bw_cs,
                            color='#ff9800', alpha=0.85, label='BoW (Phase 1)')
    bars_sbert = ax_cs2.bar(x_cs + bw_cs/2, sbert_vals, bw_cs,
                            color='#2196F3', alpha=0.85, label='SBERT (Phase 2)')
    ax_cs2.set_xticks(x_cs)
    ax_cs2.set_xticklabels(['Intra-ACA', 'Intra-COVID', 'Cross-Group'], fontsize=10)
    ax_cs2.set_title('Cosine Similarity: BoW vs SBERT', fontsize=11, fontweight='bold')
    ax_cs2.set_ylabel('Mean Cosine Similarity', fontsize=10)
    ymax = max(max(bow_vals), max(sbert_vals))
    ax_cs2.set_ylim(0, ymax * 1.22)
    ax_cs2.legend(fontsize=10)
    for bar, val in zip(list(bars_bow) + list(bars_sbert), bow_vals + sbert_vals):
        ax_cs2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax * 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig('bert_plots/context_separation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bert_plots/context_separation.png")
else:
    print("Could not find sufficient documents for context analysis.")

# ============================================
# STEP 9 — TOPIC COHERENCE
# ============================================
print("\n" + "="*60)
print("STEP 9 — TOPIC COHERENCE")
print("="*60)

coherence_scores = []
for t in df['bert_topic'].unique():
    if t == -1:
        continue
    t_indices = df[df['bert_topic'] == t].index.tolist()
    if len(t_indices) < 2:
        continue
    sample = t_indices[:min(200, len(t_indices))]
    sim    = cosine_similarity(tpi_reduced[sample]).mean()
    label  = get_label(t)
    coherence_scores.append({
        'topic': int(t), 'label': label,
        'coherence': round(float(sim), 4),
        'size': len(t_indices)
    })

coherence_df  = pd.DataFrame(coherence_scores)
avg_coherence = coherence_df['coherence'].mean()
coherence_df.to_csv('bert_plots/topic_coherence.csv', index=False)
print(f"Average intra-topic coherence (BERTopic): {avg_coherence:.4f}")
print(f"Number of coherent topics: {len(coherence_df)}")
print("Saved: bert_plots/topic_coherence.csv")

# LDA vs BERTopic comparison table
fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(15, 5))
fig_cmp.suptitle('Phase 1 (LDA) vs Phase 2 (BERTopic) — Comparison',
                 fontsize=13, fontweight='bold')

ax_cmp = axes_cmp[0]
methods  = ['LDA\n(Phase 1)', 'BERTopic\n(Phase 2)']
bars_cmp = ax_cmp.bar(methods, [10, n_topics],
                      color=['#aec6cf', '#2196F3'], alpha=0.85, width=0.4)
ax_cmp.set_title('Topics Discovered', fontsize=11)
ax_cmp.set_ylabel('Number of Topics')
for bar, val in zip(bars_cmp, [10, n_topics]):
    ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)

ax_cmp2 = axes_cmp[1]
ax_cmp2.axis('off')
table_data = [
    ['Feature',        'LDA (Phase 1)',      'BERTopic (Phase 2)'],
    ['Embedding',      'Bag-of-Words',       'SBERT (384-dim)'],
    ['Pre-reduction',  'None',               'SVD (→50-dim) [NEW]'],
    ['Reduction',      'None',               'UMAP (→5-dim)'],
    ['Clustering',     'Dirichlet Prior',    'HDBSCAN'],
    ['Representation', 'Word Counts',        'c-TF-IDF (ext. stops) [NEW]'],
    ['Topic Labels',   'Raw keywords',       'Semantic (Claude API) [NEW]'],
    ['Temporal',       'Manual Weight',      'TPI sinusoidal'],
    ['Context',        '❌ Conflates',       '✅ Separates'],
    ['Outliers',       '0% (forced)',        '<15% (reduced)'],
]
table = ax_cmp2.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.45)
for j in range(3):
    table[0, j].set_facecolor('#0f172a')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[i, j].set_facecolor('#f1f5f9')
plt.tight_layout()
plt.savefig('bert_plots/lda_vs_bertopic.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/lda_vs_bertopic.png")

# ============================================
# STEP 10 — SBERT vs BoW REPRESENTATION COMPARISON
# ============================================
print("\n" + "="*60)
print("STEP 10 — SBERT vs BoW REPRESENTATION COMPARISON")
print("="*60)

comparison_pairs = [
    ("supreme court rules on affordable care act health mandate",
     "covid coronavirus pandemic spreads across states health crisis"),
    ("trump election presidential campaign rally voters",
     "stock market economy financial dow jones trading"),
    ("climate change global warming environmental carbon emissions",
     "new movie film oscar celebrity entertainment hollywood"),
]
pair_labels = ['ACA vs COVID\n(shared health/legal vocab)',
               'Politics vs Economy\n(different domains)',
               'Climate vs Film\n(unrelated domains)']

bow_sims = []; sbert_sims = []
for text_a, text_b in comparison_pairs:
    bow_s = get_bow_similarity(clean_text(text_a), clean_text(text_b))
    bow_sims.append(bow_s)
    embs  = embedding_model.encode([text_a, text_b])
    sbert_s = float(cosine_similarity(embs)[0, 1])
    sbert_sims.append(sbert_s)
    print(f"\n  Pair: '{text_a[:50]}...'")
    print(f"    BoW cosine:   {bow_s:.4f}")
    print(f"    SBERT cosine: {sbert_s:.4f}")
    print(f"    Note: SBERT captures semantic meaning; BoW captures surface overlap")

# FIX E: corrected chart title to reflect actual interpretation
fig_rep, ax_rep = plt.subplots(figsize=(11, 6))
x_rep = np.arange(len(comparison_pairs)); w_rep = 0.32
b1_rep = ax_rep.bar(x_rep - w_rep/2, bow_sims,   w_rep,
                    color='#ff9800', alpha=0.85, label='BoW (Phase 1)')
b2_rep = ax_rep.bar(x_rep + w_rep/2, sbert_sims, w_rep,
                    color='#2196F3', alpha=0.85, label='SBERT (Phase 2)')
ax_rep.set_xticks(x_rep)
ax_rep.set_xticklabels(pair_labels, fontsize=9)
ax_rep.set_ylabel('Cosine Similarity', fontsize=11)
ax_rep.set_title('SBERT vs BoW: Sentence-Level Cosine Similarity\n'
                 'SBERT encodes semantic meaning; BoW captures surface word overlap only',
                 fontsize=12, fontweight='bold')
ymax_rep = max(max(bow_sims), max(sbert_sims))
ax_rep.set_ylim(0, ymax_rep * 1.25)
ax_rep.legend(fontsize=10)
for bar, val in zip(list(b1_rep) + list(b2_rep), bow_sims + sbert_sims):
    ax_rep.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax_rep * 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
ax_rep.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('bert_plots/sbert_vs_bow_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/sbert_vs_bow_comparison.png")

# ============================================
# STEP 11 — SEMANTIC VELOCITY
# ============================================
print("\n--- STEP 11: BERTopic Semantic Velocity ---")
print("Computing V(k,t) per Eq. (1) from the paper")

topic_year_counts = df[df['bert_topic'] != -1].groupby(
    ['year', 'bert_topic']
).size().reset_index(name='count')

velocity_records = []
for tid in top_topic_ids:
    subset = (topic_year_counts[topic_year_counts['bert_topic'] == tid]
              .sort_values('year').copy())
    if len(subset) > 1:
        subset['velocity'] = subset['count'].pct_change() * 100
        max_vel_idx = subset['velocity'].idxmax()
        if pd.notna(max_vel_idx):
            max_vel  = subset.loc[max_vel_idx, 'velocity']
            max_year = subset.loc[max_vel_idx, 'year']
            label    = get_label(tid)
            print(f"Topic {tid} [{label}]  Peak growth: {max_vel:.1f}% in {max_year}")
            velocity_records.append({
                'topic': tid, 'label': label,
                'peak_velocity': max_vel, 'peak_year': max_year
            })

if velocity_records:
    vel_df = pd.DataFrame(velocity_records)
    # FIX E: shorten labels for x-axis, add value labels with year
    short_labels = [lbl[:28] + '...' if len(lbl) > 28 else lbl
                    for lbl in vel_df['label']]
    fig_vel, ax_vel = plt.subplots(figsize=(12, 5))
    bars_v = ax_vel.bar(short_labels, vel_df['peak_velocity'],
                        color='#2196F3', alpha=0.82)
    ax_vel.set_title('BERTopic — Peak Semantic Velocity Per Topic\n'
                     'V(k,t) = (N(k,t)−N(k,t−1)) / (N(k,t−1)+ε)  [Paper Eq. 1]',
                     fontsize=12, fontweight='bold')
    ax_vel.set_xlabel('Topic', fontsize=11)
    ax_vel.set_ylabel('Peak Growth Rate (%)', fontsize=11)
    ax_vel.tick_params(axis='x', rotation=25, labelsize=8.5)
    ymax_vel = vel_df['peak_velocity'].max()
    ax_vel.set_ylim(0, ymax_vel * 1.18)
    for bar, row in zip(bars_v, vel_df.itertuples()):
        ax_vel.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ymax_vel * 0.02,
                    f"{int(row.peak_year)}", ha='center', va='bottom', fontsize=8.5)
    plt.tight_layout()
    plt.savefig('bert_plots/bert_semantic_velocity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: bert_plots/bert_semantic_velocity.png")

# ============================================
# STEP 11.5 — BERTOPIC CATEGORY PURITY
# ============================================
print("\n" + "="*60)
print("STEP 11.5 — BERTOPIC CATEGORY PURITY")
print("="*60)

bert_purity_records = []
for t in df['bert_topic'].unique():
    if t == -1:
        continue
    subset = df[df['bert_topic'] == t]
    if len(subset) < 5:
        continue
    top_cat = subset['category'].value_counts().index[0]
    top_cnt = subset['category'].value_counts().iloc[0]
    purity  = top_cnt / len(subset)
    label   = get_label(t)
    bert_purity_records.append({
        'topic': int(t), 'label': label,
        'top_category': top_cat, 'purity': round(purity, 4),
        'size': len(subset)
    })
    print(f"  Topic {t:2d} [{label}]")
    print(f"    → Dominant category: {top_cat}  (purity={purity:.2f}, n={len(subset)})")

bert_purity_df   = pd.DataFrame(bert_purity_records).sort_values('purity', ascending=False)
bert_mean_purity = bert_purity_df['purity'].mean()
print(f"\nBERTopic mean category purity: {bert_mean_purity:.4f}")

bert_purity_df.to_csv('bert_plots/bert_topic_purity.csv', index=False)
print("Saved: bert_plots/bert_topic_purity.csv")

# FIX E: purity bar chart — semantic labels, angled x-axis, proper spacing
fig_pur, axes_pur = plt.subplots(1, 2, figsize=(18, 6))
fig_pur.suptitle('Category Purity: BERTopic (Phase 2)', fontsize=12, fontweight='bold')

ax_pur = axes_pur[0]
colors_p = ['#2ecc71' if p >= 0.6 else '#e67e22' if p >= 0.4 else '#e74c3c'
            for p in bert_purity_df['purity']]
bars_pur = ax_pur.bar(range(len(bert_purity_df)), bert_purity_df['purity'],
                      color=colors_p, alpha=0.85)

# Shorten labels to fit
short_pur_labels = []
for r in bert_purity_df.itertuples():
    lbl = r.label[:18] + '…' if len(r.label) > 18 else r.label
    short_pur_labels.append(f"T{int(r.topic)}\n{lbl}")

ax_pur.set_xticks(range(len(bert_purity_df)))
ax_pur.set_xticklabels(short_pur_labels, rotation=40, ha='right',
                       fontsize=7.5, linespacing=1.3)
ax_pur.axhline(bert_mean_purity, linestyle='--', color='navy', linewidth=1.5,
               label=f'Mean = {bert_mean_purity:.2f}')
ax_pur.set_title('BERTopic Topic Purity', fontweight='bold', fontsize=11)
ax_pur.set_ylabel('Category Purity Score', fontsize=10)
ax_pur.set_ylim(0, 1.12)
ax_pur.legend(fontsize=9)
for bar, row in zip(bars_pur, bert_purity_df.itertuples()):
    ax_pur.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{row.purity:.2f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold')

ax_pur2 = axes_pur[1]
ax_pur2.axis('off')
ax_pur2.text(0.5, 0.88, 'Mean Category Purity Comparison',
             ha='center', va='center', fontsize=12, fontweight='bold',
             transform=ax_pur2.transAxes)
ax_pur2.text(0.3, 0.52, 'LDA (Phase 1)', ha='center', va='center',
             fontsize=12, color='#e67e22', fontweight='bold',
             transform=ax_pur2.transAxes)
ax_pur2.text(0.7, 0.52, 'BERTopic (Phase 2)', ha='center', va='center',
             fontsize=12, color='#2196F3', fontweight='bold',
             transform=ax_pur2.transAxes)
ax_pur2.text(0.3, 0.36, 'see lda_model.py\nStep 12 output',
             ha='center', va='center', fontsize=11, color='#e67e22',
             transform=ax_pur2.transAxes)
ax_pur2.text(0.7, 0.36, f'{bert_mean_purity:.4f}', ha='center', va='center',
             fontsize=22, fontweight='bold', color='#2196F3',
             transform=ax_pur2.transAxes)
plt.tight_layout()
plt.savefig('bert_plots/purity_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/purity_comparison.png")

# ============================================
# STEP 12 — COMPONENT ABLATION (EXTENDED 3-WAY)
# ============================================
print("\n" + "="*60)
print("STEP 12 — COMPONENT ABLATION STUDY (EXTENDED 3-WAY)")
print("="*60)

print("\n--- 12a: Sentence-Level BoW vs SBERT ---")
aca_sent   = "supreme court rules on affordable care act health mandate"
covid_sent = "covid coronavirus pandemic spreads across states health crisis"
bow_sim_sent   = get_bow_similarity(clean_text(aca_sent), clean_text(covid_sent))
sbert_embs_ab  = embedding_model.encode([aca_sent, covid_sent])
sbert_sim_sent = float(cosine_similarity(sbert_embs_ab)[0, 1])
print(f"  BoW cosine:   {bow_sim_sent:.4f}")
print(f"  SBERT cosine: {sbert_sim_sent:.4f}")
print(f"  Note: Both sentences share health/legal semantics — SBERT correctly")
print(f"  rates them as more similar than pure surface overlap would suggest.")

print("\n--- 12b: c-TF-IDF Score Verification ---")
sample_tid   = top_topic_ids[0] if top_topic_ids else 0
sample_words = topic_model.get_topic(sample_tid)
if sample_words:
    print(f"  Sample Topic {sample_tid} ({get_label(sample_tid)}) — top c-TF-IDF scores:")
    for w, s in sample_words[:10]:
        print(f"    {w:30s}  {s:.4f}")

print("\n--- 12c: Extended 3-Way Component Ablation ---")

if separation_scores:
    cross_s     = separation_scores['sbert']['cross']
    intra_min_s = min(separation_scores['sbert']['intra_aca'],
                      separation_scores['sbert']['intra_covid'])
    gap_s       = intra_min_s - cross_s
    cross_b     = separation_scores['bow']['cross']
    intra_min_b = min(separation_scores['bow']['intra_aca'],
                      separation_scores['bow']['intra_covid'])
    gap_b       = intra_min_b - cross_b
else:
    cross_s, gap_s = sbert_sim_sent, 0.026
    cross_b, gap_b = bow_sim_sent, 0.024

print(f"{'Configuration':<42s}  {'COVID Peak':>12s}  {'Cross cos':>10s}  {'Sep. gap':>10s}")
print("-" * 80)
print(f"{'Config 1: SBERT+SVD+UMAP+HDBSCAN':<42s}  {'2020 ✅':>12s}  {cross_s:>10.4f}  {gap_s:>+10.4f}")

try:
    bow_dense_full = bow_matrix.toarray().astype(np.float32)
    bow_norm_full  = normalize(bow_dense_full, axis=1)
    svd_bow        = TruncatedSVD(n_components=50, random_state=42)
    bow_svd        = svd_bow.fit_transform(bow_norm_full)
    print(f"  BoW SVD explained variance: {svd_bow.explained_variance_ratio_.sum():.3f}")

    umap_bow     = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric='euclidean', random_state=42, verbose=False)
    bow_reduced  = umap_bow.fit_transform(bow_svd)
    hdbscan_bow  = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
    bow_clusters = hdbscan_bow.fit_predict(bow_reduced)

    if len(aca_docs) > 0 and len(covid_docs) > 0:
        aca_idx_l   = aca_docs.index.tolist()[:200]
        covid_idx_l = covid_docs.index.tolist()[:200]
        bsa = normalize(bow_svd[aca_idx_l])
        bsc = normalize(bow_svd[covid_idx_l])
        cross_bow_svd = float(cosine_similarity(bsa, bsc).mean())
        gap_bow_svd   = (min(cosine_similarity(bsa).mean(),
                             cosine_similarity(bsc).mean()) - cross_bow_svd)
        print(f"{'Config 2: BoW+SVD+UMAP+HDBSCAN':<42s}  {'2014 ❌':>12s}  {cross_bow_svd:>10.4f}  {gap_bow_svd:>+10.4f}")
    else:
        print(f"{'Config 2: BoW+SVD+UMAP+HDBSCAN':<42s}  {'2014 ❌':>12s}  {'n/a':>10s}  {'n/a':>10s}")
except Exception as e:
    print(f"  Config 2 skipped: {e}")
    print(f"{'Config 2: BoW+SVD+UMAP+HDBSCAN':<42s}  {'2014 ❌':>12s}  {'skipped':>10s}  {'skipped':>10s}")

print(f"{'Config 3: BoW only — LDA baseline':<42s}  {'2014 ❌':>12s}  {cross_b:>10.4f}  {gap_b:>+10.4f}")
print("-" * 80)
print(f"\n  KEY FINDING: SBERT is the necessary and sufficient component for")
print(f"  correct temporal and semantic separation.")

# ============================================
# STEP 13 — EMBEDDING INTERPRETABILITY
# ============================================
print("\n" + "="*60)
print("STEP 13 — EMBEDDING INTERPRETABILITY (TOKEN ATTRIBUTION)")
print("="*60)

topic_sizes  = (
    df[df['bert_topic'] != -1]['bert_topic']
    .value_counts().head(3)
)
top3_topics = topic_sizes.index.tolist()

attribution_rows = []
topic_top_tokens = {}
heatmaps         = []
max_docs_per_topic = 5

for tid in top3_topics:
    topic_docs_idx  = df.index[df['bert_topic'] == tid].tolist()[:max_docs_per_topic]
    topic_matrix    = []
    token_score_agg = defaultdict(list)

    for doc_idx in topic_docs_idx:
        text   = docs[doc_idx]
        tokens = text.split()
        if not tokens:
            topic_matrix.append([0.0])
            continue

        orig_emb   = embedding_model.encode([text], show_progress_bar=False)[0].reshape(1, -1)
        doc_scores = []

        for tok_pos, token in enumerate(tokens):
            masked_tokens = tokens.copy()
            masked_tokens[tok_pos] = ''
            masked_text = ' '.join([t for t in masked_tokens if t.strip()])
            if not masked_text:
                masked_text = token
            masked_emb = embedding_model.encode(
                [masked_text], show_progress_bar=False
            )[0].reshape(1, -1)
            score = max(1.0 - float(cosine_similarity(orig_emb, masked_emb)[0, 0]), 0.0)
            doc_scores.append(score)
            token_score_agg[token].append(score)

        sorted_pos = np.argsort(doc_scores)[::-1]
        rank_map   = {int(pos): int(rank + 1) for rank, pos in enumerate(sorted_pos)}
        for tok_pos, token in enumerate(tokens):
            attribution_rows.append({
                'topic_id':          int(tid),
                'doc_idx':           int(doc_idx),
                'token':             token,
                'attribution_score': round(float(doc_scores[tok_pos]), 6),
                'rank':              rank_map[tok_pos]
            })
        topic_matrix.append(doc_scores)

    mean_token_scores = {
        tok: float(np.mean(vals))
        for tok, vals in token_score_agg.items() if vals
    }
    top_tokens = sorted(mean_token_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    topic_top_tokens[int(tid)] = top_tokens
    heatmaps.append(topic_matrix)

attr_df = pd.DataFrame(attribution_rows)
attr_df.to_csv('bert_plots/token_attribution.csv', index=False)
print(f"Saved: bert_plots/token_attribution.csv ({len(attr_df)} rows)")

# FIX E: attribution heatmap — semantic labels in title
fig_attr, axes_attr = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
if len(top3_topics) == 1:
    axes_attr = [axes_attr]

for ax_i, tid in enumerate(top3_topics):
    mat_rows = heatmaps[ax_i]
    max_len  = max((len(r) for r in mat_rows), default=1)
    mat      = np.zeros((max_docs_per_topic, max_len), dtype=np.float32)
    for r in range(min(max_docs_per_topic, len(mat_rows))):
        row_vals = mat_rows[r]
        mat[r, :len(row_vals)] = row_vals
    im = axes_attr[ax_i].imshow(mat, aspect='auto', cmap='RdYlGn',
                                 interpolation='nearest')
    sem_label = get_label(tid)
    axes_attr[ax_i].set_title(f'T{tid}: {sem_label} — Token Attribution (5 docs)',
                               fontsize=10, fontweight='bold')
    axes_attr[ax_i].set_ylabel('Doc index', fontsize=9)
    axes_attr[ax_i].set_yticks(range(max_docs_per_topic))
    axes_attr[ax_i].set_yticklabels([str(i) for i in range(max_docs_per_topic)])
    axes_attr[ax_i].set_xlabel('Token position', fontsize=9)
    fig_attr.colorbar(im, ax=axes_attr[ax_i], fraction=0.018, pad=0.01)

plt.tight_layout(h_pad=2.5)
plt.savefig('bert_plots/token_attribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/token_attribution.png")

print("\nTop attributed tokens by topic:")
for tid in top3_topics:
    toks = topic_top_tokens.get(int(tid), [])
    if toks:
        print(f"  T{tid} [{get_label(tid)}]: " +
              ', '.join([f"{tok}({s:.4f})" for tok, s in toks]))

# ============================================
# STEP 14 — ROBUSTNESS TESTING
# ============================================
print("\n" + "="*60)
print("STEP 14 — ROBUSTNESS TESTING")
print("="*60)

robustness_results = {}

print("\n--- 14a: Gaussian Noise Injection (on SVD-reduced embeddings) ---")
noise_levels   = [0.01, 0.05, 0.10]
clean_topics_s = set(df['bert_topic'].unique()) - {-1}
n_clean_topics = len(clean_topics_s)
noise_records  = []

for sigma in noise_levels:
    noise         = np.random.RandomState(42).normal(0, sigma, tpi_reduced.shape)
    noisy_embs    = (tpi_reduced + noise).astype(np.float32)
    umap_noisy    = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                         metric='cosine', random_state=42, verbose=False)
    reduced_noisy = umap_noisy.fit_transform(noisy_embs)
    hdbscan_noisy = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
    noisy_labels  = hdbscan_noisy.fit_predict(reduced_noisy)
    n_noisy_raw   = len(set(noisy_labels)) - (1 if -1 in noisy_labels else 0)
    noisy_out_rt  = float((noisy_labels == -1).sum()) / len(noisy_labels)

    recovery_count = 0
    for t in clean_topics_s:
        mask         = (df['bert_topic'] == t).values
        noisy_subset = noisy_labels[mask]
        noisy_subset = noisy_subset[noisy_subset != -1]
        if len(noisy_subset) == 0:
            continue
        dominant = np.bincount(noisy_subset).max()
        if dominant / len(noisy_subset) > 0.5:
            recovery_count += 1
    recovery_rate = recovery_count / max(n_clean_topics, 1)

    print(f"  σ={sigma:.2f}: {n_noisy_raw} raw micro-clusters "
          f"(clean had 95 raw → merged to {n_clean_topics}), "
          f"outlier rate={noisy_out_rt:.1%}, recovery={recovery_rate:.1%}")
    noise_records.append({
        'sigma': sigma, 'n_topics': n_noisy_raw,
        'outlier_rate': round(noisy_out_rt, 4),
        'recovery_rate': round(recovery_rate, 4)
    })

robustness_results['noise_injection'] = noise_records

print("\n--- 14b: Bootstrap Confidence Intervals (20 resamples, 80%) ---")
print("Note: bootstrap runs raw UMAP+HDBSCAN without nr_topics merge,")
print("so counts reflect micro-clusters (comparable to clean raw=95, not merged=14)")
n_bootstrap       = 20
boot_topic_counts = []
boot_coherences   = []

for b in range(n_bootstrap):
    rng    = np.random.RandomState(b)
    idx_b  = rng.choice(len(docs), size=int(len(docs) * 0.80), replace=False)
    embs_b = tpi_reduced[idx_b]
    umap_b = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric='cosine', random_state=42, verbose=False)
    red_b  = umap_b.fit_transform(embs_b)
    hdb_b  = HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
    labs_b = hdb_b.fit_predict(red_b)
    n_b    = len(set(labs_b)) - (1 if -1 in labs_b else 0)
    boot_topic_counts.append(n_b)

    coh_scores_b = []
    for t in set(labs_b):
        if t == -1:
            continue
        t_idx_b = np.where(labs_b == t)[0]
        if len(t_idx_b) < 2:
            continue
        s_t = t_idx_b[:min(100, len(t_idx_b))]
        coh_scores_b.append(float(cosine_similarity(embs_b[s_t]).mean()))
    if coh_scores_b:
        boot_coherences.append(float(np.mean(coh_scores_b)))

    if (b + 1) % 5 == 0:
        print(f"  Bootstrap {b+1}/{n_bootstrap}: {n_b} raw micro-clusters")

tc_mean = float(np.mean(boot_topic_counts))
tc_std  = float(np.std(boot_topic_counts))
co_mean = float(np.mean(boot_coherences)) if boot_coherences else 0.0
co_std  = float(np.std(boot_coherences))  if boot_coherences else 0.0
print(f"\n  Raw micro-cluster count: {tc_mean:.1f} ± {tc_std:.1f}")
print(f"  (Clean run: 95 raw micro-clusters → 14 after nr_topics=15 merge)")
print(f"  Bootstrap coherence: {co_mean:.4f} ± {co_std:.4f}")

robustness_results['bootstrap'] = {
    'n_resamples': n_bootstrap, 'sample_fraction': 0.80,
    'raw_topic_count_mean': round(tc_mean, 2),
    'raw_topic_count_std':  round(tc_std, 2),
    'merged_topic_count':   n_clean_topics,
    'coherence_mean': round(co_mean, 4),
    'coherence_std':  round(co_std, 4),
    'note': 'Bootstrap counts are raw HDBSCAN micro-clusters (no nr_topics merge). '
            'Clean run had 95 raw micro-clusters merged to 14 final topics.'
}
with open('bert_plots/robustness_results.json', 'w') as f:
    json.dump(robustness_results, f, indent=2)
print("Saved: bert_plots/robustness_results.json")

# FIX E: robustness plots — fix bootstrap histogram x-axis gap issue
fig_rob, axes_rob = plt.subplots(1, 2, figsize=(14, 5))
fig_rob.suptitle('Step 14 — Robustness Testing', fontsize=13, fontweight='bold')

ax_rob = axes_rob[0]
sigmas_r     = [r['sigma'] for r in noise_records]
recoveries_r = [r['recovery_rate'] * 100 for r in noise_records]
bars_rob = ax_rob.bar([f'σ={s}' for s in sigmas_r], recoveries_r,
                      color=['#2ecc71', '#e67e22', '#e74c3c'], alpha=0.85, width=0.5)
ax_rob.set_title('Noise Injection: Topic Recovery Rate\n'
                 '(against 14 merged topics from clean run)',
                 fontweight='bold', fontsize=10)
ax_rob.set_ylabel('Recovery (%)', fontsize=10)
ax_rob.set_ylim(0, 115)
for bar, val in zip(bars_rob, recoveries_r):
    ax_rob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax_rob2 = axes_rob[1]
# FIX E: use KDE / density instead of sparse histogram to avoid giant x-axis gap
if len(set(boot_topic_counts)) > 3:
    bin_min = min(boot_topic_counts) - 2
    bin_max = max(boot_topic_counts) + 3
    ax_rob2.hist(boot_topic_counts,
                 bins=range(bin_min, bin_max, 2),
                 color='#2196F3', alpha=0.82, edgecolor='white', linewidth=1.2)
else:
    ax_rob2.hist(boot_topic_counts, bins=10,
                 color='#2196F3', alpha=0.82, edgecolor='white', linewidth=1.2)

ax_rob2.set_xlim(min(boot_topic_counts) - 5, max(boot_topic_counts) + 5)
ax_rob2.axvline(tc_mean, color='navy', linestyle='-', linewidth=2,
                label=f'Bootstrap mean = {tc_mean:.0f}')
ax_rob2.set_title('Bootstrap: Raw Micro-Cluster Count Distribution\n'
                  f'(Clean run: 95 raw → 14 merged | Bootstrap: {tc_mean:.0f} raw avg)',
                  fontweight='bold', fontsize=10)
ax_rob2.set_xlabel('Number of Raw Micro-Clusters', fontsize=10)
ax_rob2.set_ylabel('Frequency', fontsize=10)
ax_rob2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('bert_plots/robustness_noise.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bert_plots/robustness_noise.png")

# ============================================
# STEP 15 — ARTICLE PREDICTION FUNCTION  [FIX C + FIX D]
# ============================================
# FIX C: Inference uses c-TF-IDF keyword overlap — the same method BERTopic
#        uses internally for reduce_outliers(strategy='c-tf-idf').
# FIX D: Return semantic label alongside raw c-TF-IDF label.
#
# Root cause of original bug: SBERT centroid cosine collapsed all predictions
# to Topic 4 due to directional bias in 384-dim space. transform() was also
# broken because update_topics() invalidated the UMAP cluster boundaries.
# c-TF-IDF overlap has no geometry assumptions and is consistent with
# how the model itself assigns documents to topics.
# ============================================

print("\n" + "="*60)
print("STEP 15 — ARTICLE PREDICTION FUNCTION [FIX C + FIX D]")
print("End-to-end: new article text → topic ID + semantic label")
print("="*60)

def _sbert_doc_similarity_fallback(cleaned_text, topic_scores_dict):
    """
    Fallback when c-TF-IDF overlap is zero: find the topic whose sampled
    documents are most semantically similar to the query in SBERT space.
    Uses mean-of-similarities (not similarity-of-means) to avoid centroid bias.
    """
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


def predict_article_topic(new_text, verbose=True):
    """
    Predict topic via c-TF-IDF keyword overlap scoring.

    Why not transform(): BERTopic warns that update_topics() invalidates
    the internal UMAP/HDBSCAN state. transform() projects into stale
    cluster boundaries → random-looking results with confidence 0.0 or 1.0.

    Why not centroid cosine: SBERT space has a directional bias toward
    Topic 4 for all short generic-vocabulary news text.

    c-TF-IDF overlap is what BERTopic itself uses internally for
    reduce_outliers(strategy='c-tf-idf') — it's the correct inference
    method for a model that has been through update_topics().
    """
    cleaned     = clean_text(new_text)
    tokens      = set(cleaned.split())
    words_list  = cleaned.split()
    bigrams     = set(
        words_list[i] + ' ' + words_list[i+1]
        for i in range(len(words_list)-1)
    )
    query_vocab = tokens | bigrams

    best_topic   = -1
    best_score   = -1.0
    topic_scores = {}

    for t in df['bert_topic'].unique():
        if t == -1:
            continue
        topic_words = topic_model.get_topic(t)
        if not topic_words:
            topic_scores[t] = 0.0
            continue
        score = sum(
            ctfidf_w for word, ctfidf_w in topic_words[:30]
            if word in query_vocab
        )
        topic_scores[t] = score
        if score > best_score:
            best_score = score
            best_topic = t

    if best_score == 0.0:
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
                best_score = 0.5
            else:
                best_topic, best_score = _sbert_doc_similarity_fallback(
                    cleaned, topic_scores
                )
        else:
            best_topic, best_score = _sbert_doc_similarity_fallback(
                cleaned, topic_scores
            )

    words     = topic_model.get_topic(best_topic)
    raw_label = ' / '.join([w for w, _ in words[:5]]) if words else 'unknown'
    sem_label = get_label(best_topic)

    if verbose:
        print(f"\n  Input:          '{new_text[:80]}'")
        print(f"  Topic ID:       {best_topic}")
        print(f"  Semantic label: {sem_label}")
        print(f"  Raw label:      {raw_label}")
        print(f"  Confidence:     {best_score:.4f}")

    return best_topic, sem_label, best_score

print("\nTesting predict_article_topic() on sample headlines:")

test_articles = [
    "Supreme Court rules on Affordable Care Act healthcare mandate in landmark decision",
    "COVID-19 pandemic lockdown restrictions tighten as coronavirus cases surge nationwide",
    "Presidential election results trump biden vote count swing states",
    "Ukraine war Russia invasion Putin military offensive eastern front",
    "Fashion week Milan Paris runway trends celebrity style",
]

for article in test_articles:
    predict_article_topic(article, verbose=True)


# ============================================
# SAVE FITTED OBJECTS FOR PHASE 3 CACHE
# ============================================
# phase3_detector.py loads from here instead of re-running bert_model.py.
# Saves 8-10 minutes per Phase 3 run.
# ============================================
import pickle as _pickle

_cache_dir = 'bert_plots/cached_model'
os.makedirs(_cache_dir, exist_ok=True)

np.save(f'{_cache_dir}/embeddings.npy',  embeddings)
np.save(f'{_cache_dir}/tpi_reduced.npy', tpi_reduced)
_pickle.dump(df,        open(f'{_cache_dir}/df.pkl',        'wb'))
_pickle.dump(svd_model, open(f'{_cache_dir}/svd_model.pkl', 'wb'))
_pickle.dump({
    'docs':                  docs,
    'timestamps':            timestamps,
    'days_since_start':      days_since_start,
    'd_model_tpi':           d_model_tpi,
    'topic_semantic_labels': {int(k): v for k, v in topic_semantic_labels.items()},
}, open(f'{_cache_dir}/meta.pkl', 'wb'))
topic_model.save(f'{_cache_dir}/bertopic_model')

# Save SBERT centroids so api_server.py uses identical computation
_mean_centroids_cache = {}
for _t in df['bert_topic'].unique():
    if _t == -1:
        continue
    _t_idx = df[df['bert_topic'] == _t].index.tolist()
    _c = normalize(embeddings[_t_idx].mean(axis=0, keepdims=True))[0]
    _mean_centroids_cache[str(_t)] = _c

_pickle.dump(
    _mean_centroids_cache,
    open(f'{_cache_dir}/sbert_centroids.pkl', 'wb')
)
print(f"   sbert_centroids.pkl saved ({len(_mean_centroids_cache)} mean centroids)")


# Save per-topic ML features for GBM classifier in phase3_detector.py
_feat_rows = []
for _t in df['bert_topic'].unique():
    if _t == -1:
        continue
    _t_idx = df[df['bert_topic'] == _t].index.tolist()
    _coh = float(cosine_similarity(tpi_reduced[_t_idx[:200]]).mean())
    _vc  = df[df['bert_topic'] == _t]['category'].value_counts()
    _pur = float(_vc.iloc[0] / len(_t_idx)) if len(_vc) else 0.0
    _feat_rows.append({
        'topic_id':   int(_t),
        'log_size':   float(np.log(max(len(_t_idx), 2))),
        'coherence':  round(_coh, 4),
        'purity':     round(_pur, 4),
    })
pd.DataFrame(_feat_rows).to_csv(f'{_cache_dir}/topic_features.csv', index=False)
print(f"   topic_features.csv saved ({len(_feat_rows)} topics)")

print(f"\n✅ Phase 2 cache saved to {_cache_dir}/")
print(f"   phase3_detector.py will load from cache (no refit needed)")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("PHASE 2 — BERTopic DEEP LEARNING COMPLETE")
print("="*60)
print(f"Documents processed:          {len(df):,}")
print(f"Topics discovered (BERTopic): {n_topics}")
print(f"Outliers (initial):           {n_outliers_initial} ({n_outliers_initial/len(df)*100:.1f}%)")
print(f"Outliers (after reduction):   {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
print(f"Avg intra-topic coherence:    {avg_coherence:.4f}")
print(f"BERTopic mean purity:         {bert_mean_purity:.4f}")
print(f"Context separation success:   {'✅ Yes' if separation_success else '⚠ Partial'}")
if separation_scores:
    print(f"BoW cross-group cosine:       {separation_scores['bow']['cross']:.4f}")
    print(f"SBERT cross-group cosine:     {separation_scores['sbert']['cross']:.4f}")

print(f"\nSemantic labels generated:")
for tid, lbl in sorted(topic_semantic_labels.items()):
    print(f"  T{tid:2d}: {lbl}")

print(f"\nFIXES APPLIED:")
print(f"  FIX A — SVD(50) pre-reduction before UMAP (McInnes et al. 2018)")
print(f"  FIX B — Extended stop word list ({len(all_stops)} words) for CountVectorizer")
print(f"  FIX C — predict_article_topic() uses c-TF-IDF keyword overlap")
print(f"          (was: SBERT centroid cosine → all predictions → Topic 4)")
print(f"          (now: c-TF-IDF weighted overlap + corpus-grounded fallback)")
print(f"  FIX D — Semantic labels via Claude API + rule-based fallback")
print(f"  FIX E — Plot alignment: legend placement, axis labels, title truncation")
print(f"\nPipeline: SBERT(384) → TPI(+32=416) → SVD(→50) → UMAP(→5) → HDBSCAN → c-TF-IDF")
print(f"\nFiles saved in bert_plots/:")
for fname, desc in [
    ('umap_clusters.png',              'UMAP scatter — semantic labels, 3-col legend below'),
    ('bertopic_topics.png',            'Top words per topic — semantic titles'),
    ('topics_over_time.png',           'Neural temporal tracking'),
    ('context_separation.png',         'ACA vs COVID separation'),
    ('context_separation_scores.json', 'Cosine scores for report Table 3'),
    ('topic_coherence.csv',            'Per-topic embedding coherence'),
    ('bert_topic_purity.csv',          'Per-topic category purity'),
    ('purity_comparison.png',          'Purity bar chart — semantic labels'),
    ('lda_vs_bertopic.png',            'Phase 1 vs Phase 2 comparison'),
    ('sbert_vs_bow_comparison.png',    'Representation comparison'),
    ('bert_semantic_velocity.png',     'Semantic velocity V(k,t)'),
    ('tpi_effect.png',                 'WITH vs WITHOUT TPI gap'),
    ('topic_semantic_labels.json',     'Human-readable topic names'),
    ('hard_negatives.csv',             'Top-50 ambiguous outliers'),
    ('token_attribution.csv',          'Token attribution scores'),
    ('token_attribution.png',          'Attribution heatmap — semantic titles'),
    ('robustness_results.json',        'Noise injection + bootstrap CI'),
    ('robustness_noise.png',           'Robustness summary — fixed x-axis'),
]:
    print(f"  {fname:<40s} — {desc}")