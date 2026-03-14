# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Phase 1 - Data Loading & EDA
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Create output folder for all EDA plots
os.makedirs('eda_plots', exist_ok=True)
print("Plots will be saved to: eda_plots/")

# ============================================
# STEP 1 - LOAD DATA
# ============================================

print("Loading dataset...")

# Load JSON file
df = pd.read_json('News_Category_Dataset_v3.json', lines=True)

print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(df.head(3))

# ============================================
# STEP 2 - BASIC CLEANING
# ============================================

# Keep relevant columns
df = df[['headline', 'short_description', 'category', 'date', 'authors']].copy()

# Combine headline + description for richer text
df['text'] = df['headline'] + ' ' + df['short_description']

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date', 'text'], inplace=True)

# Sort by date oldest to newest
df = df.sort_values('date').reset_index(drop=True)

# Balanced sample across all years
df = df.groupby(df['date'].dt.year).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
).reset_index(drop=True)
print(f"Years covered: {sorted(df['date'].dt.year.unique())}")
print(f"Total articles: {len(df)}")

print(f"\nCleaned dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nCategories found: {df['category'].nunique()}")
print(df['category'].value_counts().head(10))

# ============================================
# STEP 3 - EDA PLOTS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Dynamic Trend & Event Detector — EDA Overview', 
             fontsize=16, fontweight='bold')

# Plot 1 — Articles over time (quarterly)
df['year_month'] = df['date'].dt.to_period('Q')
quarterly_counts = df.groupby('year_month').size()
axes[0,0].plot(
    quarterly_counts.index.astype(str),
    quarterly_counts.values,
    color='steelblue', linewidth=2, marker='o', markersize=4
)
axes[0,0].set_title('Articles Published Over Time (Quarterly)')
axes[0,0].set_xlabel('Quarter')
axes[0,0].set_ylabel('Number of Articles')
axes[0,0].tick_params(axis='x', rotation=45)
every_nth = 4
for n, label in enumerate(axes[0,0].xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

# Plot 2 — Top categories
cat_counts = df['category'].value_counts().head(10)
axes[0,1].barh(cat_counts.index, cat_counts.values, color='coral')
axes[0,1].set_title('Top 10 News Categories')
axes[0,1].set_xlabel('Number of Articles')

# Plot 3 — Article text length distribution
df['word_count'] = df['text'].str.split().str.len()
axes[1,0].hist(df['word_count'], bins=40, color='green', alpha=0.7)
axes[1,0].set_title('Article Text Length Distribution')
axes[1,0].set_xlabel('Word Count')
axes[1,0].set_ylabel('Frequency')
axes[1,0].axvline(
    df['word_count'].mean(), 
    color='red', linestyle='--', 
    label=f"Mean: {df['word_count'].mean():.0f} words"
)
axes[1,0].legend()

# Plot 4 — Articles by year
df['year'] = df['date'].dt.year
year_counts = df['year'].value_counts().sort_index()
axes[1,1].bar(
    year_counts.index.astype(str),
    year_counts.values,
    color='purple', alpha=0.7
)
axes[1,1].set_title('Articles by Year (Balanced Sampling)')
axes[1,1].set_xlabel('Year')
axes[1,1].set_ylabel('Count')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_plots/eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: eda_plots/eda_overview.png")

# ============================================
# STEP 4 - TEMPORAL VELOCITY
# ============================================

df['month'] = df['date'].dt.to_period('M')
monthly = df.groupby('month').size().reset_index()
monthly.columns = ['month', 'article_count']
monthly['growth_velocity'] = monthly['article_count'].pct_change() * 100
monthly['smoothed_velocity'] = monthly['growth_velocity'].rolling(
    window=3, center=True
).mean()

print("\n--- Top 5 Months With Highest Growth Velocity ---")
print(monthly.nlargest(5, 'growth_velocity')[
    ['month', 'article_count', 'growth_velocity']
].to_string())

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    monthly['month'].astype(str),
    monthly['smoothed_velocity'],
    color='red', linewidth=2, label='Smoothed Velocity'
)
ax.plot(
    monthly['month'].astype(str),
    monthly['growth_velocity'],
    color='red', linewidth=0.5, alpha=0.3, label='Raw Velocity'
)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.fill_between(
    monthly['month'].astype(str),
    monthly['smoothed_velocity'].fillna(0),
    0,
    where=(monthly['smoothed_velocity'].fillna(0) > 0),
    color='red', alpha=0.15, label='Growing Trend'
)
ax.fill_between(
    monthly['month'].astype(str),
    monthly['smoothed_velocity'].fillna(0),
    0,
    where=(monthly['smoothed_velocity'].fillna(0) < 0),
    color='blue', alpha=0.15, label='Declining Trend'
)

tick_positions = range(0, len(monthly), 12)
tick_labels = monthly['month'].astype(str).iloc[::12]
ax.set_xticks(list(tick_positions))
ax.set_xticklabels(tick_labels, rotation=45)

ax.set_title('Monthly Article Growth Velocity\n(Positive = Emerging Trend)', 
             fontsize=13)
ax.set_xlabel('Month')
ax.set_ylabel('Growth Rate (%)')
ax.legend()
plt.tight_layout()
plt.savefig('eda_plots/velocity_preview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: eda_plots/velocity_preview.png")

# ============================================
# STEP 5 - CATEGORY VELOCITY
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
    ax.plot(
        subset['quarter'].astype(str),
        subset['count'],
        marker='o', markersize=4,
        linewidth=2.5,
        label=cat,
        color=colors[i]
    )

ax.annotate(
    '2016 US Election\nPolitics Spike',
    xy=(20, 88),
    xytext=(25, 70),
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=9,
    color='darkred'
)

ax.set_title('Category Growth Over Time — Quarterly\n(Semantic Velocity by Topic)', 
             fontsize=13)
ax.set_xlabel('Quarter')
ax.set_ylabel('Article Count')
ax.legend(loc='upper left')

every_nth = 4
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_plots/category_velocity.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: eda_plots/category_velocity.png")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*50)
print("PHASE 1 - DATA LOADING & EDA COMPLETE")
print("="*50)
print(f"Total articles loaded:    {len(df):,}")
print(f"Date range:               {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Unique categories:        {df['category'].nunique()}")
print(f"Avg words per article:    {df['word_count'].mean():.1f}")
print(f"\nFiles saved in eda_plots/:")
print(f"  eda_overview.png")
print(f"  velocity_preview.png")
print(f"  category_velocity.png")
print("\nReady for LDA modeling!")