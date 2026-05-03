# Dynamic Trend & Event Detector

A machine learning system that monitors news streams, discovers emerging topic clusters, tracks how fast each topic is growing, and verifies whether detected trends are real-world events or just noise.

Built by **Mehak Jain (230079)** and **Anuj Kumar Singh (230073)** as part of the Advanced Machine Learning & Deep Learning course, 2025.

## The Problem

Every day thousands of topics trend online. Most disappear within hours. A few — pandemics, elections, wars — are real events that matter. The question is: how do you tell them apart automatically?

The core challenge: LDA placed the COVID-19 topic peak in **2014** — six years before the pandemic — because it shares vocabulary with 2014 ACA legal news. BERTopic with SBERT embeddings corrects this to **2020** by encoding sentence meaning rather than just word counts.

## Results

| Metric | Value |
|--------|-------|
| Articles processed | 11,000 (1,000/year, 2012–2022) |
| Topics discovered | 14 (fully unsupervised) |
| COVID-19 peak — LDA | 2014 ✗ |
| COVID-19 peak — BERTopic | 2020 ✓ |
| COVID-19 velocity | +1,036% year-on-year |
| ACA vs COVID similarity (BoW) | ~0.80 (conflated) |
| ACA vs COVID similarity (SBERT) | 0.079 (separated) |
| Real events confirmed | 9 (GDELT + GBM verified) |

## Project Structure

```
AML_DL/
├── src/
│   ├── eda_pipeline.py         ← EDA and data exploration
│   ├── lda_pipeline.py         ← Phase 1: LDA topic modeling
│   ├── bertopic_pipeline.py    ← Phase 2: BERTopic + SBERT + TPI
│   ├── event_detector.py       ← Phase 3: signal classification + GDELT
│   └── app.py                  ← Flask REST API (port 8000)
├── tests/
│   └── test_suite.py           ← Unit tests
├── docs/
│   ├── report.pdf
│   └── report.tex
├── bert_plots/                 ← Generated outputs + model cache
├── model_plots/                ← Generated LDA plots
├── eda_plots/                  ← Generated EDA plots
├── index.html                  ← Interactive UI (React, no build step)
├── requirements.txt
└── README.md
```

## Setup

**Step 1 — Clone and install**

```bash
git clone https://github.com/Mehak261124/AML_DL.git
cd AML_DL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 2 — Download the dataset**

This project uses the [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle.

```bash
kaggle datasets download -d rmisra/news-category-dataset
unzip news-category-dataset.zip
```

Or download manually and place `News_Category_Dataset_v3.json` in the project root.

## Running

Run the phases in order. After the first full run, only `app.py` needs to be restarted.

```bash
# Phase 1 — Exploratory analysis (~30 seconds)
python3 src/eda_pipeline.py

# Phase 1 — LDA modeling (~3-5 minutes)
python3 src/lda_pipeline.py

# Phase 2 — BERTopic (run once, ~8-10 minutes, saves cache)
python3 src/bertopic_pipeline.py

# Phase 3 — Signal classification + GDELT verification (~2-3 minutes)
python3 src/event_detector.py

# API server — loads from cache, ready in ~30 seconds
python3 src/app.py
```

Open `http://localhost:8000` in your browser.

## Live Predict

The UI has seven pages. The **Live Predict** tab takes any news headline and returns the predicted topic, confidence score, top 3 candidate topics, a real/fake event verdict, and the five most similar articles from the corpus.

Headlines about topics outside the training corpus (AI research, climate policy, sports) return **No Matching Topic Found** with an UNCERTAIN verdict rather than a wrong answer.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "COVID-19 vaccine rollout begins across the country"}'
```

## How It Works

**Phase 1 — LDA baseline**
TF-IDF vectorization followed by LDA with three novel temporal features: logarithmic recency weight, category velocity score, and text richness ratio. Establishes the bag-of-words conflation problem as motivation for Phase 2.

**Phase 2 — BERTopic**
SBERT (all-MiniLM-L6-v2, 384-dim) encodes full sentence meaning. We introduce **Temporal Positional Injection (TPI)** — a novel contribution that appends 32-dim sinusoidal time encodings to each embedding before manifold learning, improving the ACA-COVID separation gap from 0.157 to 0.184. The full pipeline: SBERT → TPI (416-dim) → SVD (50-dim) → UMAP (5-dim) → HDBSCAN → c-TF-IDF.

**Phase 3 — Event detection**
BERTrend popularity metric classifies each topic-year as NOISE, WEAK, STRONG, or EMERGING. EMERGING events are cross-verified against GDELT DOC 2.0. A GBM classifier trained on topic coherence, purity, log-size, and velocity provides a P(real event) probability score for every prediction.

## Tests

```bash
python -m pytest tests/test_suite.py -v
```

Covers text cleaning, feature engineering, TPI encoding, BoW similarity, GBM verdict mapping, and API route definitions. Tests requiring a trained model skip automatically if the cache is absent.

## Tech Stack

`sentence-transformers` · `bertopic` · `umap-learn` · `hdbscan` · `scikit-learn` · `flask` · `pandas` · `numpy` · `matplotlib` · React 18

## References

Blei et al. (2003) · Grootendorst (2022) · Reimers & Gurevych (2019) · McInnes et al. (2018) · Campello et al. (2013) · Vaswani et al. (2017) · Boutaleb et al. (2024) · Leetaru & Schrodt (2013)