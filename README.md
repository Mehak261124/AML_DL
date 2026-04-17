# Dynamic Trend & Event Detector
## Problem Statement

How do you distinguish a fleeting social media meme from a significant real-world news event — automatically, in real time?

This project builds a **Dynamic Trend & Event Detector** that monitors news streams, groups articles into topics, tracks how fast each topic is growing over time (semantic velocity), and verifies detected trends against real-world news sources (GDELT).

---

## Key Results

| Metric | Value |
|---|---|
| Dataset | HuffPost News (2012–2022) |
| Articles Processed | 11,000 (balanced across 11 years) |
| Topics Discovered | 10 (fully unsupervised) |
| Trump & President topic peak growth | 278% in 2014 |
| COVID-19 topic detected | Automatically in 2020 |
| Weighted LDA Confidence | 0.6103 |

---

## Project Structure

```
AML_DL/
├── eda_plots/                  ← EDA visualizations
│   ├── eda_overview.png
│   ├── velocity_preview.png
│   └── category_velocity.png
├── model_plots/                ← Model results & feature plots
│   ├── baseline_tfidf.png
│   ├── feature_engineering.png
│   ├── confidence_comparison.png
│   ├── lda_topics.png
│   ├── lda_topic_evolution.png
│   └── semantic_velocity.png
├── main.py                     ← EDA pipeline
├── lda_model.py                ← LDA modeling pipeline
├── requirements.txt            ← All dependencies
└── News_Category_Dataset_v3.json  ← Dataset (download separately)
```

---

## Setup Instructions

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AML_DL.git
cd AML_DL
```

---

### Step 2 — Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal.

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages including pandas, scikit-learn, BERTopic, sentence-transformers, UMAP, and HDBSCAN.

> **Note:** Installation takes 5–10 minutes due to large ML packages.

---

### Step 4 — Download the Dataset

This project uses the **HuffPost News Category Dataset** from Kaggle.

#### Option A — Kaggle API (Recommended)

**4a. Get your Kaggle API key:**
1. Go to [kaggle.com](https://kaggle.com) → Profile → Account
2. Scroll to **API** section → Click **"Create New Token"**
3. A file `kaggle.json` downloads automatically

**4b. Place the API key:**
```bash
# Mac/Linux
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\
```

**4c. Download the dataset:**
```bash
kaggle datasets download -d rmisra/news-category-dataset
```

**4d. Unzip the dataset:**
```bash
# Mac/Linux
unzip news-category-dataset.zip

# Windows
tar -xf news-category-dataset.zip
```

**4e. Verify the file exists:**
```bash
ls -lh News_Category_Dataset_v3.json
# Should show ~83MB file
```

#### Option B — Manual Download

1. Go to: [https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
2. Click **Download**
3. Unzip the file
4. Place `News_Category_Dataset_v3.json` in the `AML_DL/` folder

---

### Step 5 — Verify Setup

```bash
# Check Python version (need 3.8+)
python3 --version

# Check key packages installed
python3 -c "import pandas; import sklearn; import bertopic; print('All packages OK')"
```

---

## Running the Project

### Run 1 — EDA Pipeline (`main.py`)

**What it does:**
- Loads and cleans 11,000 articles (1,000 per year, 2012–2022)
- Generates 3 EDA visualizations saved to `eda_plots/`
- Shows article distribution, top categories, word count distribution
- Calculates monthly growth velocity per category

```bash
python3 main.py
```

**Expected output:**
```
Plots will be saved to: eda_plots/
Loading dataset...
Years covered: [2012, 2013, 2014, ..., 2022]
Total articles: 11000
Saved: eda_plots/eda_overview.png
Saved: eda_plots/velocity_preview.png
Saved: eda_plots/category_velocity.png
PHASE 1 - DATA LOADING & EDA COMPLETE
```

**Expected runtime:** ~30 seconds

**Output plots:**

| Plot | Description |
|---|---|
| `eda_overview.png` | 4-panel overview: articles over time, categories, word count, yearly distribution |
| `velocity_preview.png` | Monthly growth velocity with smoothing |
| `category_velocity.png` | Per-category growth trends (quarterly) |

---

### Run 2 — LDA Modeling Pipeline (`lda_model.py`)

**What it does:**
- Runs TF-IDF baseline topic extraction
- Fits LDA with 10 topics on count vectors
- Applies 3 novel feature engineering techniques
- Compares standard vs temporally-weighted LDA
- Performs temporal topic tracking and semantic velocity analysis
- Runs failure analysis on the COVID/Court topic conflation

```bash
python3 lda_model.py
```

**Expected output:**
```
Plots will be saved to: model_plots/
Loading data...
Total documents: 11000
--- BASELINE: TF-IDF Frequency Extraction ---
Vocabulary size: 5000
--- ADVANCED ML: LDA Topic Modeling ---
Fitting LDA with 10 topics...
LDA fitting complete!
--- LDA Discovered Topics ---
Topic  0: news | said | fox | ukraine | war ...
Topic  7: trump | president | donald | trumps ...
Topic  8: covid | court | coronavirus | state ...
FEATURE ENGINEERING — NOVEL CONTRIBUTIONS
Standard LDA avg confidence:        0.6104
Temporally-Weighted LDA confidence: 0.6103
PHASE 1 - LDA MODELING COMPLETE
```

**Expected runtime:** 3–5 minutes (LDA fitting is compute-intensive)

**Output plots:**

| Plot | Description |
|---|---|
| `baseline_tfidf.png` | Top 20 words by TF-IDF score |
| `feature_engineering.png` | 3 novel features: temporal weight, category velocity, text richness |
| `confidence_comparison.png` | Standard LDA vs Temporally-Weighted LDA |
| `lda_topics.png` | Top words per topic (all 10 topics) |
| `lda_topic_evolution.png` | Topic growth over time 2012–2022 |
| `semantic_velocity.png` | Peak growth rate per topic with year annotation |

---

## Approach & Methodology

### Phase 1 — Advanced ML (This Repository)

```
Raw Text
    ↓
Text Cleaning (lowercase, remove punctuation)
    ↓
Baseline: TF-IDF Vectorization
    ↓
Advanced ML: LDA Topic Modeling (10 topics)
    ↓
Novel Feature Engineering:
  • Temporal Weight (log recency)
  • Category Velocity Score
  • Text Richness Ratio
    ↓
Temporally-Weighted LDA
    ↓
Semantic Velocity Calculation
    ↓
Failure Analysis
```

### Phase 2 — Deep Learning (Implemented)

```
SBERT Embeddings (all-MiniLM-L6-v2, 384-dim)
    ↓
UMAP Dimensionality Reduction (384 → 5 dim)
    ↓
HDBSCAN Density Clustering (auto topic count)
    ↓
c-TF-IDF Topic Representation
    ↓
Context Separation Analysis
    ↓
Interactive React UI + Flask API
```

**Key Results:**
- 15 topics discovered (auto, vs LDA's fixed 10)
- COVID-19 topic correctly peaks in 2020 (1,036% velocity)
- Context separation ✅ — ACA legal ≠ COVID pandemic
- SBERT cosine similarity: 0.27 (vs BoW ~0.8+)

---

## Novel Feature Engineering

Three novel features were engineered to address LDA's static corpus assumption:

**Feature 1 — Temporal Weight**
```
w(t) = log(1 + days_since_start) / max(log(1 + days_since_start))
```
Assigns higher weight to recent documents. Addresses LDA's core limitation of treating all documents as temporally equivalent.

**Feature 2 — Category Velocity Score**
Captures the growth rate of an article's category at the time of publication. Articles published during topic surges carry stronger trend signal.

**Feature 3 — Text Richness Ratio**
```
richness = unique_words / (total_words + 1)
```
Measures semantic density. Topically rich articles contribute more meaningful signal to LDA.

---

## Key Findings

**Finding 1 — Politics dominates (2012–2022)**
TF-IDF identifies "trump" as the highest scoring term across the entire corpus, reflecting US political dominance in news coverage.

**Finding 2 — LDA discovers COVID-19 unsupervised**
LDA automatically identifies a covid/court/coronavirus topic cluster without any labeling — demonstrating the model's ability to detect real-world events.

**Finding 3 — Trump topic grows 278% from 2013–2016**
Corresponding exactly to the US presidential campaign period.

**Finding 4 — Failure Analysis — COVID/Court conflation**
The covid/court topic shows peak velocity in 2014, predating COVID-19. This occurs because LDA's bag-of-words assumption conflates legal news (court, state, health) with pandemic news — same vocabulary, different semantics. This motivates Phase 2's SBERT approach.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF, LDA, CountVectorizer |
| `matplotlib / seaborn` | Visualizations |
| `scipy` | Sparse matrix operations for temporal weighting |
| `BERTopic` | Phase 2 — deep learning topic modeling |
| `sentence-transformers` | Phase 2 — SBERT embeddings |
| `umap-learn` | Phase 2 — dimensionality reduction |
| `hdbscan` | Phase 2 — density-based clustering |
| `kaggle` | Dataset download |

---

## Dataset

**HuffPost News Category Dataset**
- Source: Kaggle — [rmisra/news-category-dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- Size: 210,000+ articles (full), 11,000 used (balanced subset)
- Date range: 2012–2022
- Categories: 42 news categories
- Citation: Misra, R. (2022). News Category Dataset.

> The full dataset (8.8GB) is identified for future GPU-enabled experiments. Current methodology is designed to scale directly to the full dataset without architectural changes.

---

## References

- Blei, D., Ng, A., & Jordan, M. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993–1022.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv:1802.03426*.
- Misra, R. (2022). News Category Dataset. *Kaggle*.
- Allan, J., et al. (1998). Topic Detection and Tracking Pilot Study. *DARPA*.