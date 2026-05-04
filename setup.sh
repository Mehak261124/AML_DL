#!/usr/bin/env bash
# setup.sh — Turn-key setup for Dynamic Trend & Event Detector
# Usage: bash setup.sh
set -e

echo "=================================================="
echo "  Dynamic Trend & Event Detector — Setup Script"
echo "=================================================="

# 1. Python version check
python3 --version || { echo "Python 3 required"; exit 1; }

# 2. Virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
fi
echo "[1/4] Activating venv..."
source venv/bin/activate

# 3. Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "      Done."

# 4. Dataset check
echo "[3/4] Checking dataset..."
if [ ! -f "News_Category_Dataset_v3.json" ]; then
    echo "      Dataset not found. Downloading via Kaggle CLI..."
    if command -v kaggle &>/dev/null; then
        kaggle datasets download -d rmisra/news-category-dataset --unzip
    else
        echo ""
        echo "      kaggle CLI not installed. Download manually:"
        echo "      https://www.kaggle.com/datasets/rmisra/news-category-dataset"
        echo "      Place News_Category_Dataset_v3.json in the project root."
    fi
else
    echo "      Dataset found."
fi

# 5. Architecture diagram
echo "[4/4] Generating architecture diagram..."
python3 src/generate_architecture.py

echo ""
echo "=================================================="
echo "  Setup complete! Run the pipeline in order:"
echo ""
echo "    python3 src/eda_pipeline.py        # ~30 s"
echo "    python3 src/lda_model.py           # ~3 min"
echo "    python3 src/bert_model.py          # ~8 min (saves cache)"
echo "    python3 src/event_detector.py      # ~2 min"
echo "    python3 src/app.py                 # starts API on :8000"
echo ""
echo "  Then open http://localhost:8000 in your browser."
echo "=================================================="
