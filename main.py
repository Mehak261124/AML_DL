# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Phase 1 — Step A: Data Loading & EDA
# ============================================
#
# References:
#   Blei, D., Ng, A., Jordan, M. (2003). Latent Dirichlet
#       Allocation. JMLR 3, 993-1022.
#   Misra, R. (2022). News Category Dataset. Kaggle.
#       https://kaggle.com/datasets/rmisra/news-category-dataset
#
# Note on train/val/test split:
#   This pipeline is fully unsupervised (LDA, BERTopic).
#   There is no prediction target and therefore no
#   generalisation loss to measure. A held-out split is
#   not applicable; instead, we validate topic quality
#   via semantic coherence (Phase 2 Step 9) and
#   category-alignment checks (Phase 1 Step 11).
# ============================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — required for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['eda', 'lda', 'bert', 'all'], default='all')
args = parser.parse_args()

if args.phase not in ('eda', 'all'):
    print(f"Skipping EDA in main.py for phase='{args.phase}'.")
    sys.exit(0)

os.makedirs('eda_plots', exist_ok=True)
print("EDA plots will be saved to: eda_plots/")

# ============================================
# STEP 1 — LOAD DATA
# ============================================

print("\n" + "="*60)
print("PHASE 1 — DATA LOADING & EDA")
print("="*60)
print("\nLoading dataset...")

df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
print(f"Raw dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ============================================
# STEP 2 — CLEANING & BALANCED SAMPLING
# ============================================
# Rationale for stratified 1,000/year sampling:
#   Election years (2016, 2020) contain significantly more
#   political articles, which would inflate temporal velocity
#   metrics if left uncontrolled. Strict per-year capping
#   removes this volume bias before any modelling.

df = df[['headline', 'short_description', 'category', 'date', 'authors']].copy()
df['text']  = df['headline'] + ' ' + df['short_description']
df['date']  = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)
df = df.sort_values('date').reset_index(drop=True)

df = df.groupby(df['date'].dt.year).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)

df['year']       = df['date'].dt.year
df['word_count'] = df['text'].str.split().str.len()

print(f"\nCleaned & balanced dataset shape: {df.shape}")
print(f"Date range:   {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Years covered: {sorted(df['year'].unique())}")
print(f"Unique categories: {df['category'].nunique()}")
print(f"\nTop categories:\n{df['category'].value_counts().head(10).to_string()}")

# ============================================
# STEP 3 — EDA VISUALISATIONS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Dynamic Trend & Event Detector — EDA Overview\n'
             '(HuffPost News 2012–2022, n=11,000 stratified)',
             fontsize=15, fontweight='bold')

# --- Plot 1: Articles over time (quarterly) ---
df['year_quarter'] = df['date'].dt.to_period('Q')
quarterly = df.groupby('year_quarter').size()
axes[0,0].plot(quarterly.index.astype(str), quarterly.values,
               color='steelblue', linewidth=2, marker='o', markersize=4)
axes[0,0].set_title('Articles Published Over Time (Quarterly)', fontweight='bold')
axes[0,0].set_xlabel('Quarter')
axes[0,0].set_ylabel('Article Count')
axes[0,0].tick_params(axis='x', rotation=45)
for n, label in enumerate(axes[0,0].xaxis.get_ticklabels()):
    if n % 4 != 0:
        label.set_visible(False)
axes[0,0].grid(alpha=0.3)

# --- Plot 2: Top 10 categories ---
cat_counts = df['category'].value_counts().head(10)
axes[0,1].barh(cat_counts.index, cat_counts.values, color='coral', alpha=0.85)
axes[0,1].set_title('Top 10 News Categories', fontweight='bold')
axes[0,1].set_xlabel('Article Count')
axes[0,1].grid(axis='x', alpha=0.3)

# --- Plot 3: Word count distribution ---
axes[1,0].hist(df['word_count'], bins=40, color='seagreen', alpha=0.75, edgecolor='white')
axes[1,0].set_title('Article Text Length Distribution\n(headline + description)',
                    fontweight='bold')
axes[1,0].set_xlabel('Word Count')
axes[1,0].set_ylabel('Frequency')
mu = df['word_count'].mean()
axes[1,0].axvline(mu, color='crimson', linestyle='--', linewidth=2,
                  label=f'Mean: {mu:.0f} words')
axes[1,0].legend()
axes[1,0].grid(alpha=0.3)

# --- Plot 4: Articles by year ---
year_counts = df['year'].value_counts().sort_index()
axes[1,1].bar(year_counts.index.astype(str), year_counts.values,
              color='mediumpurple', alpha=0.8, edgecolor='white')
axes[1,1].set_title('Articles by Year\n(Stratified Balanced Sampling: 1,000/year)',
                    fontweight='bold')
axes[1,1].set_xlabel('Year')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(axis='y', alpha=0.3)
# Annotate to confirm strict 1,000-cap per year
for i, (yr, cnt) in enumerate(year_counts.items()):
    axes[1,1].text(i, cnt + 5, str(cnt), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('eda_plots/eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: eda_plots/eda_overview.png")

# ============================================
# STEP 4 — TEMPORAL VELOCITY (MONTHLY)
# ============================================
# Semantic velocity V(k,t) = (N(k,t) - N(k,t-1)) / (N(k,t-1) + ε)
# Here we compute corpus-level velocity before any topic model
# to verify that the signal exists in raw article counts.

df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').size().reset_index()
monthly.columns = ['month', 'article_count']
monthly['growth_velocity']   = monthly['article_count'].pct_change() * 100
monthly['smoothed_velocity'] = monthly['growth_velocity'].rolling(
    window=3, center=True, min_periods=1
).mean()

print("\n--- Top 5 Months by Growth Velocity ---")
top5 = monthly.nlargest(5, 'growth_velocity')[
    ['month', 'article_count', 'growth_velocity']
]
print(top5.to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 5))
months_str = monthly['month'].astype(str)
ax.plot(months_str, monthly['smoothed_velocity'],
        color='crimson', linewidth=2, label='3-month Smoothed Velocity')
ax.plot(months_str, monthly['growth_velocity'],
        color='crimson', linewidth=0.5, alpha=0.3, label='Raw Velocity')
ax.axhline(0, color='black', linestyle='--', alpha=0.4)
ax.fill_between(months_str, monthly['smoothed_velocity'].fillna(0), 0,
                where=monthly['smoothed_velocity'].fillna(0) > 0,
                color='red', alpha=0.12, label='Positive Velocity')
ax.fill_between(months_str, monthly['smoothed_velocity'].fillna(0), 0,
                where=monthly['smoothed_velocity'].fillna(0) < 0,
                color='blue', alpha=0.12, label='Negative Velocity')
ticks = range(0, len(monthly), 12)
ax.set_xticks(list(ticks))
ax.set_xticklabels(monthly['month'].astype(str).iloc[::12], rotation=45)
ax.set_title('Monthly Article Growth Velocity\n'
             'V(t) = (N(t) − N(t−1)) / N(t−1)  —  positive = emerging trend',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Growth Rate (%)')
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('eda_plots/velocity_preview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/velocity_preview.png")

# ============================================
# STEP 5 — CATEGORY-LEVEL VELOCITY
# ============================================

cat_time = df.groupby(
    [df['date'].dt.to_period('Q'), 'category']
).size().reset_index()
cat_time.columns = ['quarter', 'category', 'count']
top_cats = df['category'].value_counts().head(5).index.tolist()
cat_time_top = cat_time[cat_time['category'].isin(top_cats)]

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, cat in enumerate(top_cats):
    subset = cat_time_top[cat_time_top['category'] == cat].copy()
    ax.plot(subset['quarter'].astype(str), subset['count'],
            marker='o', markersize=4, linewidth=2.5,
            label=cat, color=colors[i])

# Annotate the well-documented 2016 US election spike
ax.annotate('2016 US Election\nPolitics Spike',
            xy=(20, 88), xytext=(25, 72),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            fontsize=9, color='darkred', fontweight='bold')

ax.set_title('Category Growth Over Time — Quarterly\n'
             '(Semantic Velocity by Category — pre-topic-model sanity check)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Quarter')
ax.set_ylabel('Article Count')
ax.legend(loc='upper left', fontsize=9)
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % 4 != 0:
        label.set_visible(False)
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('eda_plots/category_velocity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_plots/category_velocity.png")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("PHASE 1 — DATA LOADING & EDA COMPLETE")
print("="*60)
print(f"Total articles:        {len(df):,}")
print(f"Date range:            {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Unique categories:     {df['category'].nunique()}")
print(f"Mean words/article:    {df['word_count'].mean():.1f}")
print(f"\nFiles saved in eda_plots/:")
print(f"  eda_overview.png      — 4-panel EDA overview")
print(f"  velocity_preview.png  — Monthly article growth velocity")
print(f"  category_velocity.png — Per-category quarterly velocity")
print("\nReady for Phase 1 LDA modelling (lda_model.py)")
