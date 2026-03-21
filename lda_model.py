# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Phase 1 - Baseline + LDA Topic Modeling
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from scipy.sparse import diags
import re

# Create output folder for all plots
os.makedirs('model_plots', exist_ok=True)
print("Plots will be saved to: model_plots/")

# ============================================
# STEP 1 - RELOAD DATA
# ============================================

print("\nLoading data...")
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
df = df[['headline', 'short_description', 'category', 'date']].copy()
df['text'] = df['headline'] + ' ' + df['short_description']
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)
df = df.sort_values('date').reset_index(drop=True)

# Balanced sample across years
df = df.groupby(df['date'].dt.year).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)

print(f"Total documents: {len(df)}")
print(f"Years covered: {sorted(df['date'].dt.year.unique())}")

# ============================================
# STEP 2 - TEXT CLEANING
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['word_count'] = df['clean_text'].str.split().str.len()
print("Text cleaning done.")

# ============================================
# STEP 3 - BASELINE MODEL (TF-IDF)
# ============================================

print("\n--- BASELINE: TF-IDF Frequency Extraction ---")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    max_df=0.95,
    min_df=2
)
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
feature_names = tfidf.get_feature_names_out()

# Top words overall
mean_tfidf = tfidf_matrix.mean(axis=0).A1
top_indices = mean_tfidf.argsort()[-20:][::-1]
top_words = [feature_names[i] for i in top_indices]

print(f"Vocabulary size: {len(feature_names)}")
print(f"Top 20 words overall: {top_words}")

# Visualize baseline top words
plt.figure(figsize=(12, 5))
plt.bar(top_words, mean_tfidf[top_indices], color='steelblue', alpha=0.8)
plt.title('Baseline: Top 20 Words by TF-IDF Score', fontsize=13)
plt.xlabel('Word')
plt.ylabel('Mean TF-IDF Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('model_plots/baseline_tfidf.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_plots/baseline_tfidf.png")

# ============================================
# STEP 4 - ADVANCED ML: LDA (STANDARD)
# ============================================

print("\n--- ADVANCED ML: LDA Topic Modeling ---")

# LDA needs COUNT vectors (not TF-IDF)
count_vectorizer = CountVectorizer(
    max_features=5000,
    stop_words='english',
    max_df=0.95,
    min_df=2
)
count_matrix = count_vectorizer.fit_transform(df['clean_text'])
count_feature_names = count_vectorizer.get_feature_names_out()

print(f"Document-term matrix shape: {count_matrix.shape}")

# Fit LDA with 10 topics
N_TOPICS = 10
print(f"\nFitting LDA with {N_TOPICS} topics...")
print("(Takes 2-3 minutes — normal!)")

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=20,
    learning_method='batch',
    verbose=1
)
lda.fit(count_matrix)
print("LDA fitting complete!")

# ============================================
# STEP 5 - SHOW LDA TOPICS
# ============================================

print("\n--- LDA Discovered Topics ---")
topic_labels = []
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words_lda = [count_feature_names[i] for i in top_words_idx]
    topic_labels.append(top_words_lda[0])
    print(f"Topic {topic_idx:2d}: {' | '.join(top_words_lda)}")

# ============================================
# STEP 6 - ASSIGN TOPICS TO DOCUMENTS
# ============================================

doc_topic_dist = lda.transform(count_matrix)
df['lda_topic'] = doc_topic_dist.argmax(axis=1)
df['lda_confidence'] = doc_topic_dist.max(axis=1)

print(f"\nTopic assignment complete!")
print(f"Average confidence: {df['lda_confidence'].mean():.3f}")
print(f"\nTopic distribution:")
print(df['lda_topic'].value_counts().sort_index())

# ============================================
# STEP 7 - FEATURE ENGINEERING (NOVEL)
# ============================================

print("\n" + "="*50)
print("FEATURE ENGINEERING — NOVEL CONTRIBUTIONS")
print("="*50)

# --------------------------------------------------
# Feature 1: Temporal Weight
# Standard LDA treats 2012 and 2022 articles equally.
# We assign higher weight to recent articles using
# logarithmic scaling of days since corpus start.
# Motivation: Recent documents carry stronger trend signal.
# --------------------------------------------------
df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
df['temporal_weight'] = np.log1p(df['days_from_start'])
df['temporal_weight'] = df['temporal_weight'] / df['temporal_weight'].max()

print(f"\nFeature 1 — Temporal Weight:")
print(f"  Range: {df['temporal_weight'].min():.3f} to {df['temporal_weight'].max():.3f}")
print(f"  Oldest article weight: {df['temporal_weight'].min():.3f}")
print(f"  Newest article weight: {df['temporal_weight'].max():.3f}")

# --------------------------------------------------
# Feature 2: Category Velocity Score
# Captures how fast the article's category was growing
# at the time of publication.
# Motivation: Articles published during topic surges
# carry stronger trend signal than baseline coverage.
# --------------------------------------------------
df['month_str'] = df['date'].dt.to_period('M').astype(str)
monthly_cat = df.groupby(
    ['month_str', 'category']
).size().reset_index()
monthly_cat.columns = ['month_str', 'category', 'cat_count']
monthly_cat['cat_velocity'] = monthly_cat.groupby(
    'category'
)['cat_count'].pct_change().fillna(0)

df = df.merge(
    monthly_cat[['month_str', 'category', 'cat_velocity']],
    on=['month_str', 'category'],
    how='left'
)
df['cat_velocity'] = df['cat_velocity'].fillna(0)

print(f"\nFeature 2 — Category Velocity Score:")
print(f"  Range: {df['cat_velocity'].min():.3f} to {df['cat_velocity'].max():.3f}")
print(f"  Mean velocity: {df['cat_velocity'].mean():.3f}")
print(f"  Articles with positive velocity: {(df['cat_velocity'] > 0).sum():,}")

# --------------------------------------------------
# Feature 3: Text Richness Ratio
# Ratio of unique words to total words.
# Motivation: Topically rich articles have higher
# unique word density — more informative for LDA.
# --------------------------------------------------
df['unique_words'] = df['clean_text'].apply(
    lambda x: len(set(x.split()))
)
df['text_richness'] = df['unique_words'] / (df['word_count'] + 1)

print(f"\nFeature 3 — Text Richness Ratio:")
print(f"  Range: {df['text_richness'].min():.3f} to {df['text_richness'].max():.3f}")
print(f"  Mean richness: {df['text_richness'].mean():.3f}")

# --------------------------------------------------
# Visualize all 3 features
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Novel Feature Engineering\n(Addressing LDA\'s Static Assumption)',
             fontsize=14, fontweight='bold')

axes[0].hist(df['temporal_weight'], bins=40, color='steelblue', alpha=0.8)
axes[0].set_title('Feature 1: Temporal Weight\n(log recency — recent = higher weight)')
axes[0].set_xlabel('Weight Score (0=oldest, 1=newest)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['temporal_weight'].mean(), color='red',
                linestyle='--', label=f"Mean: {df['temporal_weight'].mean():.2f}")
axes[0].legend()

axes[1].hist(df['cat_velocity'].clip(-1, 2), bins=40, color='coral', alpha=0.8)
axes[1].set_title('Feature 2: Category Velocity\n(topic growth rate at publication)')
axes[1].set_xlabel('Velocity Score')
axes[1].set_ylabel('Frequency')
axes[1].axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero growth')
axes[1].legend()

axes[2].hist(df['text_richness'], bins=40, color='green', alpha=0.8)
axes[2].set_title('Feature 3: Text Richness\n(unique word ratio)')
axes[2].set_xlabel('Richness Score')
axes[2].set_ylabel('Frequency')
axes[2].axvline(df['text_richness'].mean(), color='red',
                linestyle='--', label=f"Mean: {df['text_richness'].mean():.2f}")
axes[2].legend()

plt.tight_layout()
plt.savefig('model_plots/feature_engineering.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: model_plots/feature_engineering.png")

# --------------------------------------------------
# Apply Temporal Weighting to LDA input matrix
# This is the core contribution — weighted LDA
# --------------------------------------------------
print("\nApplying temporal weighting to document-term matrix...")
temporal_weights_sparse = diags(df['temporal_weight'].values)
weighted_count_matrix = temporal_weights_sparse @ count_matrix
print(f"Weighted matrix shape: {weighted_count_matrix.shape}")

# Fit Temporally-Weighted LDA
print("\nFitting Temporally-Weighted LDA...")
lda_weighted = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=20,
    learning_method='batch',
    verbose=0
)
lda_weighted.fit(weighted_count_matrix)

doc_topic_weighted = lda_weighted.transform(weighted_count_matrix)
df['lda_weighted_topic'] = doc_topic_weighted.argmax(axis=1)
df['lda_weighted_confidence'] = doc_topic_weighted.max(axis=1)

std_conf = df['lda_confidence'].mean()
wtd_conf = df['lda_weighted_confidence'].mean()
improvement = (wtd_conf - std_conf) / std_conf * 100

print(f"\n--- Confidence Comparison ---")
print(f"Standard LDA avg confidence:         {std_conf:.4f}")
print(f"Temporally-Weighted LDA confidence:  {wtd_conf:.4f}")
print(f"Improvement:                         {improvement:.2f}%")

# Visualize confidence comparison
fig, ax = plt.subplots(figsize=(8, 5))
models = ['Standard LDA\n(Baseline)', 'Temporally-Weighted LDA\n(Our Contribution)']
confidences = [std_conf, wtd_conf]
colors_bar = ['#aec6cf', '#2196F3']
bars = ax.bar(models, confidences, color=colors_bar, alpha=0.85, width=0.4)
ax.set_title('Standard LDA vs Temporally-Weighted LDA\n(Average Topic Confidence)',
             fontsize=13)
ax.set_ylabel('Average Topic Confidence')
ax.set_ylim(0, max(confidences) * 1.2)
for bar, val in zip(bars, confidences):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.annotate(f'+{improvement:.1f}% improvement',
            xy=(1, wtd_conf),
            xytext=(0.6, wtd_conf + 0.015),
            arrowprops=dict(arrowstyle='->', color='green'),
            color='green', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('model_plots/confidence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_plots/confidence_comparison.png")

# ============================================
# STEP 8 - VISUALIZE LDA TOPICS (WITH NAMES)
# ============================================

# Build topic names from top 2 words
topic_names = {}
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-3:][::-1]
    top_words_lda = [count_feature_names[i] for i in top_words_idx]
    name = ' & '.join([w.capitalize() for w in top_words_lda[:2]])
    topic_names[topic_idx] = name

print("\nTopic Names:")
for tid, name in topic_names.items():
    print(f"  Topic {tid} → {name}")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('LDA — Top Words Per Topic', fontsize=14, fontweight='bold')

for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-8:][::-1]
    top_words_lda = [count_feature_names[i] for i in top_words_idx]
    top_scores = topic[top_words_idx]
    top_scores = top_scores / top_scores.sum()

    ax = axes[topic_idx // 5][topic_idx % 5]
    ax.barh(top_words_lda[::-1], top_scores[::-1], color='coral', alpha=0.8)
    ax.set_title(f'{topic_names[topic_idx]}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Word Probability')

plt.tight_layout()
plt.savefig('model_plots/lda_topics.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_plots/lda_topics.png")

# ============================================
# STEP 9 - TEMPORAL TOPIC TRACKING
# ============================================

print("\nTracking LDA topics over time...")

df['year'] = df['date'].dt.year
topic_time = df.groupby(['year', 'lda_topic']).size().reset_index()
topic_time.columns = ['year', 'topic', 'count']

top_topics = df['lda_topic'].value_counts().head(5).index.tolist()

plt.figure(figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, tid in enumerate(top_topics):
    subset = topic_time[topic_time['topic'] == tid]
    plt.plot(
        subset['year'],
        subset['count'],
        marker='o', linewidth=2.5,
        markersize=6,
        label=f"{topic_names[tid]}",
        color=colors[i]
    )

plt.title('LDA Topic Evolution Over Time\n(Temporal Tracking)', fontsize=13)
plt.xlabel('Year')
plt.ylabel('Article Count per Topic')
plt.legend(loc='upper left', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('model_plots/lda_topic_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_plots/lda_topic_evolution.png")

# ============================================
# STEP 10 - SEMANTIC VELOCITY
# ============================================

print("\n--- LDA Semantic Velocity ---")
print("How fast is each topic growing year over year?\n")

velocity_records = []
for tid in top_topics:
    subset = topic_time[topic_time['topic'] == tid].sort_values('year').copy()
    if len(subset) > 1:
        subset['velocity'] = subset['count'].pct_change() * 100
        max_vel = subset['velocity'].max()
        max_year = subset.loc[subset['velocity'].idxmax(), 'year']
        top_words_idx = lda.components_[tid].argsort()[-3:][::-1]
        label = ' / '.join([count_feature_names[j] for j in top_words_idx])
        print(f"Topic {tid} [{label}]")
        print(f"  Peak growth: {max_vel:.1f}% in {max_year}")
        velocity_records.append({
            'topic': tid,
            'label': label,
            'peak_velocity': max_vel,
            'peak_year': max_year
        })

if velocity_records:
    vel_df = pd.DataFrame(velocity_records)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        vel_df['label'],
        vel_df['peak_velocity'],
        color='tomato', alpha=0.8
    )
    plt.title('Peak Semantic Velocity Per Topic\n(Max Year-over-Year Growth %)',
              fontsize=13)
    plt.xlabel('Topic')
    plt.ylabel('Peak Growth Rate (%)')
    plt.xticks(rotation=30, ha='right')
    for bar, row in zip(bars, vel_df.itertuples()):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{int(row.peak_year)}",
            ha='center', va='bottom', fontsize=9
        )
    plt.tight_layout()
    plt.savefig('model_plots/semantic_velocity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: model_plots/semantic_velocity.png")

# ============================================
# STEP 11 - VALIDATE AGAINST REAL CATEGORIES
# ============================================

print("\n--- LDA Topic vs Real Category Validation ---")
print("Does LDA find topics matching real news categories?\n")

for tid in top_topics[:5]:
    topic_docs = df[df['lda_topic'] == tid]
    top_cat = topic_docs['category'].value_counts().index[0]
    top_pct = (topic_docs['category'].value_counts().iloc[0]
               / len(topic_docs) * 100)
    top_words_idx = lda.components_[tid].argsort()[-4:][::-1]
    label = ' | '.join([count_feature_names[j] for j in top_words_idx])
    print(f"Topic {tid}: [{label}]")
    print(f"  → Maps to: {top_cat} ({top_pct:.1f}% match)")
    print()

# ============================================
# STEP 12 - FAILURE ANALYSIS
# ============================================

print("\n--- FAILURE ANALYSIS ---")
print("Analyzing specific cases where LDA failed\n")

# Find the Covid & Court topic (contains 'covid')
covid_topic = None
for tid, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    words = [count_feature_names[i] for i in top_words_idx]
    if 'covid' in words:
        covid_topic = tid
        break

if covid_topic is not None:
    print(f"COVID topic identified: Topic {covid_topic} ({topic_names[covid_topic]})")
    covid_docs = df[df['lda_topic'] == covid_topic]

    year_dist = covid_docs['year'].value_counts().sort_index()
    print(f"\nYear distribution of COVID topic:")
    print(year_dist.to_string())

    pre_covid = covid_docs[covid_docs['year'] < 2020]
    print(f"\nDocuments in COVID topic BEFORE 2020: {len(pre_covid)}")
    print("Sample headlines from pre-2020 'COVID' topic:")
    print(pre_covid['text'].head(5).to_string())

    print(f"\nMathematical explanation:")
    print(f"LDA's bag-of-words assumption collapses semantic context.")
    print(f"Words like 'state', 'health', 'people', 'court' appear in")
    print(f"BOTH legal news (pre-2020) AND COVID news (post-2020).")
    print(f"Cosine similarity in word space = high overlap → same topic.")
    print(f"SBERT (Phase 2) encodes full sentence meaning,")
    print(f"separating 'court ruling on health policy' from")
    print(f"'coronavirus spreading through communities'.")
else:
    print("COVID topic not found in top words — check topic list above.")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*50)
print("PHASE 1 - LDA MODELING COMPLETE")
print("="*50)
print(f"Documents processed:         {len(df):,}")
print(f"Topics discovered:           {N_TOPICS}")
print(f"Standard LDA confidence:     {std_conf:.4f}")
print(f"Weighted LDA confidence:     {wtd_conf:.4f}")
print(f"Improvement from features:   {improvement:.2f}%")
print(f"\nFiles saved in model_plots/:")
print(f"  baseline_tfidf.png")
print(f"  feature_engineering.png")
print(f"  confidence_comparison.png")
print(f"  lda_topics.png")
print(f"  lda_topic_evolution.png")
print(f"  semantic_velocity.png")
print(f"\nNext: Phase 2 — BERTopic Deep Learning")
