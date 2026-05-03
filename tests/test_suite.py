# ============================================
# PROJECT: DYNAMIC TREND & EVENT DETECTOR
# Unit Tests — Pipeline Validation
# ============================================
# Run:  python -m pytest test_pipeline.py -v
#   or: python -m unittest test_pipeline -v
# ============================================

import unittest
import re
import os
import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# Inline copies of core functions (avoids executing the full pipeline scripts
# on import, which load the 200K-article dataset and fit models).
# ---------------------------------------------------------------------------

def clean_text_bert(text):
    """BERT-style cleaning: preserves digits for SBERT."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_text_lda(text):
    """LDA-style cleaning: strips digits and special chars."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_text_richness(text):
    """R_d = |U(d)| / (|d| + 1)"""
    tokens = text.split()
    return len(set(tokens)) / (len(tokens) + 1)


def compute_temporal_weight(days_from_start, max_days):
    """w(d) = log(1 + Δt_d) / max_d log(1 + Δt_d)"""
    if max_days <= 0:
        return 0.0
    return float(np.log1p(days_from_start) / np.log1p(max_days))


def build_tpi_encoding(year_frac, d_model=32):
    """Sinusoidal temporal positional encoding (Vaswani-style)."""
    pe = np.zeros(d_model, dtype=np.float32)
    for i in range(0, d_model, 2):
        denom = 10000.0 ** (i / d_model)
        pe[i]     = np.sin(year_frac / denom)
        if i + 1 < d_model:
            pe[i + 1] = np.cos(year_frac / denom)
    return pe


# ===================================================================
# TEST CLASS 1: Text Cleaning
# ===================================================================

class TestTextCleaning(unittest.TestCase):
    """Verifies cleaning functions match expected behaviour."""

    def test_bert_preserves_digits(self):
        """BERT cleaning must preserve digits (covid 19, 2020)."""
        self.assertIn('19', clean_text_bert('COVID-19 pandemic'))
        self.assertIn('2020', clean_text_bert('Year 2020!'))

    def test_lda_strips_digits(self):
        """LDA cleaning must strip all digits."""
        result = clean_text_lda('COVID-19 pandemic 2020')
        self.assertNotIn('19', result)
        self.assertNotIn('2020', result)
        self.assertIn('covid', result)

    def test_lowercase(self):
        self.assertEqual(clean_text_bert('HELLO World'), 'hello world')
        self.assertEqual(clean_text_lda('HELLO World'), 'hello world')

    def test_special_chars_removed(self):
        result = clean_text_bert("it's a test! @#$%")
        self.assertNotIn('@', result)
        self.assertNotIn('#', result)
        self.assertNotIn("'", result)

    def test_extra_whitespace_collapsed(self):
        self.assertEqual(clean_text_bert('  hello   world  '), 'hello world')

    def test_empty_string(self):
        self.assertEqual(clean_text_bert(''), '')
        self.assertEqual(clean_text_lda(''), '')

    def test_non_string_input(self):
        """Should handle numeric or None input via str()."""
        self.assertEqual(clean_text_bert(12345), '12345')
        self.assertEqual(clean_text_lda(None), 'none')


# ===================================================================
# TEST CLASS 2: Feature Engineering
# ===================================================================

class TestFeatureEngineering(unittest.TestCase):

    def test_text_richness_unique_words(self):
        """All unique words → richness near 1.0."""
        r = compute_text_richness('apple banana cherry date elderberry')
        self.assertGreater(r, 0.8)

    def test_text_richness_repeated_words(self):
        """All same word → richness near 1/(n+1)."""
        r = compute_text_richness('the the the the the')
        self.assertLess(r, 0.3)

    def test_temporal_weight_range(self):
        """Weight must be in [0, 1]."""
        for days in [0, 100, 500, 3650]:
            w = compute_temporal_weight(days, 3650)
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_temporal_weight_monotonic(self):
        """More recent = higher weight."""
        w1 = compute_temporal_weight(100, 3650)
        w2 = compute_temporal_weight(3000, 3650)
        self.assertGreater(w2, w1)

    def test_temporal_weight_max_is_one(self):
        w = compute_temporal_weight(3650, 3650)
        self.assertAlmostEqual(w, 1.0, places=5)

    def test_temporal_weight_zero_days(self):
        w = compute_temporal_weight(0, 3650)
        self.assertAlmostEqual(w, 0.0, places=5)


# ===================================================================
# TEST CLASS 3: TPI Encoding
# ===================================================================

class TestTPIEncoding(unittest.TestCase):
    """Temporal Positional Injection must produce correct shapes."""

    def test_tpi_shape(self):
        pe = build_tpi_encoding(0.5, d_model=32)
        self.assertEqual(pe.shape, (32,))

    def test_tpi_bounded(self):
        """Sinusoidal encoding values must be in [-1, 1]."""
        pe = build_tpi_encoding(3.14, d_model=32)
        self.assertTrue(np.all(pe >= -1.0))
        self.assertTrue(np.all(pe <= 1.0))

    def test_tpi_different_times(self):
        """Different timestamps produce different encodings."""
        pe1 = build_tpi_encoding(0.0, d_model=32)
        pe2 = build_tpi_encoding(5.0, d_model=32)
        self.assertFalse(np.allclose(pe1, pe2))

    def test_tpi_augmented_dim(self):
        """384 SBERT + 32 TPI → 416-dim augmented vector."""
        sbert = np.random.randn(384).astype(np.float32)
        tpi = build_tpi_encoding(1.0, d_model=32)
        augmented = np.concatenate([sbert, tpi])
        self.assertEqual(augmented.shape[0], 416)


# ===================================================================
# TEST CLASS 4: BoW Similarity
# ===================================================================

class TestBoWSimilarity(unittest.TestCase):

    def test_identical_texts_high_similarity(self):
        """Identical cleaned texts should have cosine ≈ 1.0."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize

        vec = CountVectorizer(max_features=1000, stop_words='english')
        texts = ['supreme court health care law mandate ruling',
                 'supreme court health care law mandate ruling']
        X = vec.fit_transform(texts).toarray().astype(np.float32)
        Xn = normalize(X, axis=1)
        sim = float(Xn[0] @ Xn[1])
        self.assertGreater(sim, 0.99)

    def test_unrelated_texts_low_similarity(self):
        """Completely different texts should have low cosine."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import normalize

        vec = CountVectorizer(max_features=1000, stop_words='english')
        texts = ['supreme court health care law mandate ruling',
                 'basketball football soccer tennis sports game']
        X = vec.fit_transform(texts).toarray().astype(np.float32)
        Xn = normalize(X, axis=1)
        sim = float(Xn[0] @ Xn[1])
        self.assertLess(sim, 0.3)


# ===================================================================
# TEST CLASS 5: Output Files Validation
# ===================================================================

class TestOutputFiles(unittest.TestCase):
    """Check that expected pipeline outputs exist (if pipeline has been run)."""

    BERT_PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'bert_plots')

    @unittest.skipUnless(os.path.isdir(os.path.join(os.path.dirname(__file__), 'bert_plots')),
                         'bert_plots directory not found — run bert_model.py first')
    def test_key_plots_exist(self):
        expected = [
            'umap_clusters.png', 'bertopic_topics.png',
            'topics_over_time.png', 'context_separation.png',
            'topic_coherence.csv', 'lda_vs_bertopic.png',
            'tpi_effect.png', 'token_attribution.png',
        ]
        for name in expected:
            path = os.path.join(self.BERT_PLOTS_DIR, name)
            self.assertTrue(os.path.exists(path), f'Missing: {name}')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__),
                                    'bert_plots/context_separation_scores.json')),
        'Separation scores not found — run bert_model.py first')
    def test_separation_scores_valid_json(self):
        path = os.path.join(self.BERT_PLOTS_DIR, 'context_separation_scores.json')
        with open(path) as f:
            data = json.load(f)
        self.assertIn('bow', data)
        self.assertIn('sbert', data)
        self.assertIn('cross', data['sbert'])

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__),
                                    'bert_plots/robustness_results.json')),
        'Robustness results not found — run bert_model.py first')
    def test_robustness_results_valid(self):
        path = os.path.join(self.BERT_PLOTS_DIR, 'robustness_results.json')
        with open(path) as f:
            data = json.load(f)
        self.assertIn('noise_injection', data)
        self.assertIn('bootstrap', data)
        self.assertEqual(len(data['noise_injection']), 3)  # 3 sigma levels


# ===================================================================
# TEST CLASS 6: API Server Endpoints (mock)
# ===================================================================

class TestAPIEndpoints(unittest.TestCase):
    """Smoke-tests API endpoint definitions (does NOT require model loaded)."""

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__), 'api_server.py')),
        'api_server.py not found')
    def test_api_server_has_required_routes(self):
        """Verify route definitions exist in source code."""
        with open(os.path.join(os.path.dirname(__file__), 'api_server.py')) as f:
            source = f.read()
        required_routes = [
            '/api/predict', '/api/topics', '/api/plots/',
            '/api/summary', '/api/coherence', '/api/separation',
            '/api/purity', '/api/attribution/', '/api/attribution_live',
            '/api/similarity',
        ]
        for route in required_routes:
            self.assertIn(route, source, f'Missing route: {route}')


class TestGBMClassifier(unittest.TestCase):
    """Validates GBM real/fake classifier behaviour."""

    def _make_feature(self, vel, sc, nz, coh, pur, log_sz):
        return [[vel, sc, nz, coh, pur, log_sz]]

    def test_predict_proba_range(self):
        """predict_proba must always return a value in [0, 1]."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        X = [
            [0.0, 0, 1, 0.6, 0.5, 4.0],  # noise-like
            [10.0, 3, 3, 0.85, 0.9, 3.5], # real-like
            [0.5, 1, 2, 0.7, 0.6, 3.0],  # borderline
        ]
        y = [0, 1, 0]
        base  = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model = CalibratedClassifierCV(base, cv=2, method='isotonic')
        model.fit(X, y)
        for feat in X:
            prob = float(model.predict_proba([feat])[0][1])
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_verdict_mapping_exhaustive(self):
        """All five verdict strings must be reachable from the threshold map."""
        thresholds = [
            (0.80, 'VERIFIED_REAL'),
            (0.62, 'LIKELY_REAL'),
            (0.45, 'UNCERTAIN'),
            (0.30, 'LIKELY_NOISE'),
            (0.10, 'VERIFIED_NOISE'),
        ]
        def map_verdict(p):
            if p >= 0.75:   return 'VERIFIED_REAL'
            if p >= 0.55:   return 'LIKELY_REAL'
            if p >= 0.40:   return 'UNCERTAIN'
            if p >= 0.25:   return 'LIKELY_NOISE'
            return 'VERIFIED_NOISE'

        expected_verdicts = {v for _, v in thresholds}
        produced_verdicts = {map_verdict(p) for p, _ in thresholds}
        self.assertEqual(expected_verdicts, produced_verdicts)

    def test_high_velocity_emerging_predicts_real(self):
        """An EMERGING topic with high velocity should score higher than a NOISE topic."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        X = [
            [0.0, 0, 1, 0.5, 0.4, 4.0],
            [0.1, 0, 1, 0.55, 0.45, 3.8],
            [8.0, 3, 3, 0.88, 0.92, 3.5],
            [5.0, 3, 2, 0.82, 0.85, 3.2],
        ]
        y = [0, 0, 1, 1]
        base  = GradientBoostingClassifier(n_estimators=20, random_state=42)
        model = CalibratedClassifierCV(base, cv=2, method='isotonic')
        model.fit(X, y)
        p_noise   = float(model.predict_proba([[0.0, 0, 1, 0.5, 0.4, 4.0]])[0][1])
        p_emerging = float(model.predict_proba([[9.0, 3, 3, 0.9, 0.95, 3.5]])[0][1])
        self.assertGreater(p_emerging, p_noise)


if __name__ == '__main__':
    unittest.main()
