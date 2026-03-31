"""
02_nlp_pipeline.py
==================
Core NLP module — the heart of this project.

Produces per-meeting numeric features from raw FOMC text:

  1. Keyword-based hawkish/dovish score  (fast, interpretable, no GPU needed)
  2. FinBERT sentiment (positive / negative / neutral)  (best accuracy, needs GPU)
  3. TF-IDF change vectors              (captures vocabulary shifts)
  4. Sentence-level embedding similarity to "policy tight" / "policy easy" anchors

Outputs:
  data/fomc_nlp_features.csv  — one row per meeting, ready for modeling

Usage:
  python 02_nlp_pipeline.py [--statements data/fomc_statements.json]
                            [--minutes   data/fomc_minutes.json]
                            [--device    cpu|cuda]
                            [--no-finbert]   # skip FinBERT if no GPU / memory

Requirements:
  pip install transformers torch sentence-transformers scikit-learn pandas tqdm
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Hawkish / Dovish Lexicon ─────────────────────────────────────────────────
# Carefully hand-curated based on Fed-watcher literature:
# Loughran-McDonald finance word lists + FOMC-specific vocabulary.
# Score: +1 per hawkish word, -1 per dovish word (normalized by word count).

HAWKISH_TERMS = [
    # Rate / tightening language
    "tighten", "tightening", "restrictive", "restriction", "raise rates",
    "rate increase", "rate hike", "hike", "hikes", "higher rates",
    "less accommodative", "firming", "firm", "elevated inflation",
    # Inflation concern
    "inflation risk", "inflationary", "inflation expectations unanchored",
    "price pressures", "upside risk", "upside risks",
    "above.*target", "persistently high",
    # Labor / demand heat
    "tight labor", "overheating", "robust demand", "strong growth",
    "rapid growth", "strong job", "labor market tight",
    # Policy action words
    "vigilant", "determined", "resolve", "committed to reducing",
    "necessary to raise", "further increases",
]

DOVISH_TERMS = [
    # Rate / easing language
    "cut", "ease", "easing", "accommodative", "accommodation",
    "lower rates", "rate reduction", "rate cut", "reduce the target",
    "less restrictive", "normalize", "patient",
    # Inflation comfort
    "inflation.*below target", "subdued inflation", "low inflation",
    "well-anchored", "inflation expectations anchored",
    "disinflation", "deflationary", "downside risk", "downside risks",
    # Labor / demand weakness
    "slack", "underutilization", "elevated unemployment",
    "soft", "softening", "slowing", "slowdown", "weakness",
    "below.*potential", "spare capacity",
    # Policy pause / hold language
    "pause", "hold", "patient", "gradual", "gradual approach",
    "data-dependent", "monitoring", "cautious",
]

# Sentence-level semantic anchors for embedding similarity
HAWKISH_ANCHOR = (
    "The Federal Reserve remains determined to restore price stability "
    "and will raise interest rates further as needed."
)
DOVISH_ANCHOR = (
    "The Federal Reserve can afford to be patient and may reduce "
    "interest rates to support economic growth and employment."
)


# ── Keyword Scorer ───────────────────────────────────────────────────────────
def compile_patterns(terms: list[str]) -> list[re.Pattern]:
    """
    Compile regex patterns for hawkish/dovish term matching.
    Multi-word phrases (e.g. 'rate hike') use simple substring match;
    single words use word-boundary anchors to avoid partial matches.
    """
    patterns = []
    for t in terms:
        if " " in t:
            patterns.append(re.compile(re.escape(t), re.IGNORECASE))
        else:
            patterns.append(re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE))
    return patterns


HAWK_PATTERNS = compile_patterns(HAWKISH_TERMS)
DOVE_PATTERNS = compile_patterns(DOVISH_TERMS)


def keyword_score(text: str) -> dict:
    """
    Return raw counts and normalized hawkishness score for a document.
    hawk_score > 0 → hawkish, < 0 → dovish, ≈ 0 → neutral.
    """
    words = text.split()
    n = max(len(words), 1)

    hawk_hits = sum(bool(p.search(text)) for p in HAWK_PATTERNS)
    dove_hits = sum(bool(p.search(text)) for p in DOVE_PATTERNS)

    # Frequency-weighted version (count all occurrences, not just binary)
    hawk_freq = sum(len(p.findall(text)) for p in HAWK_PATTERNS)
    dove_freq = sum(len(p.findall(text)) for p in DOVE_PATTERNS)

    return {
        "hawk_hits": hawk_hits,
        "dove_hits": dove_hits,
        "hawk_net": hawk_hits - dove_hits,  # binary net
        "hawk_score_norm": (hawk_freq - dove_freq) / n * 100,  # per 100 words
        "hawk_ratio": hawk_hits / max(hawk_hits + dove_hits, 1),  # 0=pure dove, 1=pure hawk
        "tone_word_count": hawk_hits + dove_hits,
    }


# ── FinBERT Sentiment ────────────────────────────────────────────────────────
def load_finbert(device: str = "cpu"):
    """
    Load FinBERT from HuggingFace. Returns (tokenizer, model, pipeline).
    Model: ProsusAI/finbert — fine-tuned on financial news for
    positive / negative / neutral classification.
    """
    from transformers import pipeline as hf_pipeline

    print("Loading FinBERT (ProsusAI/finbert) …")
    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=0 if device == "cuda" else -1,
        truncation=True,
        max_length=512,
    )
    print("  FinBERT loaded.")
    return finbert


def finbert_score_document(text: str, pipe, chunk_size: int = 400) -> dict:
    """
    FOMC documents can be thousands of words — longer than FinBERT's 512-token
    context. Strategy: split into non-overlapping chunks, score each, average.

    Returns averaged logits mapped to:
      finbert_positive, finbert_negative, finbert_neutral  (0-1, sum to ~1)
      finbert_net  = positive - negative  (-1 to 1)
    """
    # Split into sentences, then pack into ~400-word chunks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        wlen = len(sent.split())
        if current_len + wlen > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = [sent], wlen
        else:
            current.append(sent)
            current_len += wlen
    if current:
        chunks.append(" ".join(current))

    if not chunks:
        return {
            "finbert_positive": np.nan,
            "finbert_negative": np.nan,
            "finbert_neutral": np.nan,
            "finbert_net": np.nan,
            "finbert_chunks": 0,
        }

    results = pipe(chunks, batch_size=8)

    label_map = {"positive": [], "negative": [], "neutral": []}
    for r in results:
        label = r["label"].lower()
        score = r["score"]
        # FinBERT returns score for the predicted label only; map to probabilities
        label_map[label].append(score)
        for other in label_map:
            if other != label:
                label_map[other].append((1 - score) / 2)

    pos = float(np.mean(label_map["positive"]))
    neg = float(np.mean(label_map["negative"]))
    neu = float(np.mean(label_map["neutral"]))

    return {
        "finbert_positive": pos,
        "finbert_negative": neg,
        "finbert_neutral": neu,
        "finbert_net": pos - neg,  # key feature: positive=dovish in finance context
        "finbert_chunks": len(chunks),
    }


# ── Sentence Embedding Similarity ────────────────────────────────────────────
def load_sentence_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load a small sentence transformer. all-MiniLM-L6-v2 is only 80MB and fast.
    For best quality, use 'sentence-transformers/all-mpnet-base-v2'.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading sentence transformer ({model_name}) …")
    model = SentenceTransformer(model_name)
    print("  Sentence transformer loaded.")
    return model


def embedding_hawkishness(text: str, sent_model, anchors: dict) -> dict:
    """
    Compute cosine similarity between the document embedding and
    hand-crafted hawkish / dovish anchor sentences.

    embedding_hawk_sim > embedding_dove_sim → hawkish lean
    """
    doc_emb = sent_model.encode([text[:3000]])  # truncate for speed
    hawk_sim = float(cosine_similarity(doc_emb, anchors["hawk"])[0][0])
    dove_sim = float(cosine_similarity(doc_emb, anchors["dove"])[0][0])
    return {
        "emb_hawk_sim": hawk_sim,
        "emb_dove_sim": dove_sim,
        "emb_hawk_net": hawk_sim - dove_sim,  # positive = hawkish
    }


# ── TF-IDF Change Features ────────────────────────────────────────────────────
def compute_tfidf_features(texts: list[str], dates: list[str]) -> pd.DataFrame:
    """
    Fit TF-IDF on the full corpus, then compute:
      - cosine similarity between consecutive meetings (language drift)
      - first principal component score (captures dominant vocabulary direction)

    Returns a DataFrame indexed by date.
    """
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    X = vec.fit_transform(texts)  # shape (n_docs, vocab)

    # Pairwise cosine similarity between adjacent meetings
    cos_with_prev = [np.nan]
    for i in range(1, X.shape[0]):
        sim = cosine_similarity(X[i], X[i - 1])[0][0]
        cos_with_prev.append(float(sim))

    # LSA: top-2 components capture macro vocabulary direction
    svd = TruncatedSVD(n_components=2, random_state=42)
    svd_scores = svd.fit_transform(X)

    return pd.DataFrame(
        {
            "date_str": dates,
            "tfidf_cos_prev": cos_with_prev,
            "tfidf_pc1": svd_scores[:, 0],
            "tfidf_pc2": svd_scores[:, 1],
        }
    )


# ── Feature Extraction Orchestrator ─────────────────────────────────────────
def extract_features(
    records: list[dict],
    doc_type: str,
    use_finbert: bool = True,
    use_embeddings: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run the full NLP pipeline on a list of FOMC text records.

    Args:
        records:        list of dicts with keys 'date_str', 'text', 'year'
        doc_type:       'statement' or 'minutes'
        use_finbert:    if False, skip FinBERT (useful without GPU)
        use_embeddings: if False, skip sentence-transformer
        device:         'cpu' or 'cuda'

    Returns: DataFrame with one row per meeting
    """
    print(f"\n{'='*60}")
    print(f"NLP pipeline for {len(records)} {doc_type} documents")
    print(f"{'='*60}")

    rows = []

    # ── Optional: load heavy models once ────────────────────────────────────
    finbert_pipe = load_finbert(device) if use_finbert else None
    sent_model = load_sentence_model() if use_embeddings else None
    anchors = {}
    if sent_model is not None:
        anchors["hawk"] = sent_model.encode([HAWKISH_ANCHOR])
        anchors["dove"] = sent_model.encode([DOVISH_ANCHOR])

    # ── Per-document features ────────────────────────────────────────────────
    for i, rec in enumerate(records):
        print(f"  [{i+1}/{len(records)}] {rec.get('date_str', '?')} …", end="", flush=True)
        text = rec.get("text", "")
        if not text.strip():
            print(" SKIP (empty)")
            continue

        row = {
            "date_str": rec.get("date_str", ""),
            "year": rec.get("year"),
            "doc_type": doc_type,
            "word_count": len(text.split()),
        }

        # 1. Keyword scores (always fast)
        kw = keyword_score(text)
        row.update(kw)

        # 2. FinBERT
        if finbert_pipe is not None:
            fb = finbert_score_document(text, finbert_pipe)
            row.update(fb)

        # 3. Sentence embedding similarity
        if sent_model is not None and anchors:
            emb = embedding_hawkishness(text, sent_model, anchors)
            row.update(emb)

        rows.append(row)
        print(" ✓")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Corpus-level TF-IDF features ────────────────────────────────────────
    print("Computing TF-IDF corpus features …")
    tfidf_df = compute_tfidf_features(
        [r["text"] for r in records if r.get("text")],
        [r.get("date_str", "") for r in records if r.get("text")],
    )
    df = df.merge(tfidf_df, on="date_str", how="left")

    # ── Derived composite features ────────────────────────────────────────────
    # Ensemble: average of available hawk signals (normalized to 0-1)
    hawk_signals = []
    if "hawk_ratio" in df:
        hawk_signals.append(df["hawk_ratio"])
    if "finbert_net" in df:
        # finbert_net in (-1,1); rescale to (0,1). Note: in finance text,
        # "positive" often means "markets are doing well / growth" which
        # can be dovish. Interpret carefully.
        hawk_signals.append((df["finbert_net"] + 1) / 2)
    if "emb_hawk_net" in df:
        hawk_signals.append((df["emb_hawk_net"] + 1) / 2)

    if hawk_signals:
        df["hawk_composite"] = pd.concat(hawk_signals, axis=1).mean(axis=1)

    # Tone change from previous meeting (momentum signal)
    df = df.sort_values("date_str").reset_index(drop=True)
    df["hawk_score_delta"] = df["hawk_score_norm"].diff()
    df["hawk_ratio_delta"] = df["hawk_ratio"].diff()

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FOMC NLP feature extraction")
    p.add_argument("--statements", default="data/fomc_statements.json")
    p.add_argument("--minutes", default="data/fomc_minutes.json")
    p.add_argument("--out", default="data/fomc_nlp_features.csv")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--no-finbert", action="store_true", help="Skip FinBERT (no GPU / memory)")
    p.add_argument("--no-embeddings", action="store_true", help="Skip sentence transformers")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_dfs = []

    for path, doc_type in [(args.statements, "statement"), (args.minutes, "minutes")]:
        p = Path(path)
        if not p.exists():
            print(f"⚠ File not found: {p} — skipping")
            continue
        records = json.loads(p.read_text())
        df = extract_features(
            records,
            doc_type=doc_type,
            use_finbert=not args.no_finbert,
            use_embeddings=not args.no_embeddings,
            device=args.device,
        )
        all_dfs.append(df)

    if not all_dfs:
        print("No data found. Run 01_collect_fomc_text.py first.")
        raise SystemExit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n✓ NLP features saved → {out_path}")
    print(combined.describe().to_string())
