# Decoding the Fed: Using NLP to Predict Yield Curve Movements from FOMC Language

**Course Project — Data Science**
**GitHub Repository:** `ds-project`

| Name | NetID | Role |
|------|-------|------|
| Charles Timmes | cwt32 | NLP pipeline, data collection |
| Raj Shah | ras637 | Modeling, yield data, evaluation |

---

## Research Question & Motivation

Can the language and tone of Federal Open Market Committee (FOMC) statements and meeting minutes help predict short-term movements in the U.S. Treasury yield curve?

The Federal Reserve plays a major role in financial markets, especially in bond markets. After each FOMC meeting, investors closely examine policy statements and meeting minutes to understand the Fed's position on inflation, employment, interest rates, and overall economic conditions. Even small wording changes can influence market expectations and move Treasury yields.

This project is interesting because the rate decision itself is not the only important signal — the language surrounding the decision may also contain useful information. More hawkish language may suggest tighter future policy, while more dovish language may suggest a softer stance. We test whether these language patterns can be measured with NLP and used to explain or predict yield movements.

This question matters from both a data science and economics perspective. It connects text analysis with real financial outcomes and applies NLP, predictive modeling, and time series analysis to a real-world market problem. Prior work in financial NLP and sentiment analysis suggests that textual tone can affect asset prices, which motivates our focus on Fed communication and Treasury market reactions.

---

## Data Sources

**FOMC Text Data**
- FOMC policy statements and meeting minutes from the [Federal Reserve website](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- Historical archive: https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm
- HuggingFace dataset (pre-cleaned): https://huggingface.co/datasets/seanchua/FOMC

**Treasury Yield & Macro Data (FRED)**
- [DGS2](https://fred.stlouisfed.org/series/DGS2) — 2-Year Treasury Constant Maturity Rate
- [DGS5](https://fred.stlouisfed.org/series/DGS5) — 5-Year Treasury Constant Maturity Rate
- [DGS10](https://fred.stlouisfed.org/series/DGS10) — 10-Year Treasury Constant Maturity Rate
- [DFF](https://fred.stlouisfed.org/series/DFF) — Federal Funds Effective Rate
- [CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL) — CPI All Urban Consumers
- UNRATE — Unemployment Rate

**NLP Model**
- [FinBERT (ProsusAI)](https://huggingface.co/ProsusAI/finbert) — BERT fine-tuned on financial news for sentiment classification

---

## Methodology

### Text Preprocessing
1. Remove vote tallies, headers, boilerplate, and implementation notes from FOMC documents
2. Tokenize and clean using Python NLP tools
3. Match each document to its correct release date
4. Align FOMC release dates with Treasury yield data
5. Create target variables based on 1-day, 3-day, and 5-day yield changes after each release

### NLP Feature Extraction (three layers)

**Layer 1 — Keyword Lexicon** (fast, interpretable, no GPU needed)

Hand-curated hawkish/dovish term lists based on Loughran-McDonald finance word lists and FOMC-specific vocabulary. Produces per-meeting hawkishness scores normalized by document length.

**Layer 2 — FinBERT Sentiment** (best accuracy, GPU recommended)

ProsusAI/finbert classifies text as positive/negative/neutral. Because FOMC documents exceed FinBERT's 512-token limit, we chunk documents into ~400-word pieces, score each chunk, and average across the document.

**Layer 3 — Sentence Embedding Similarity**

Sentence-transformers encode documents and compute cosine similarity to hand-crafted hawkish/dovish anchor sentences. Captures semantic tone beyond keyword matching.

**Corpus-level features:** TF-IDF change vectors, cosine similarity to prior meeting (language drift), LSA principal components.

### Models

We evaluate a progression of models comparing macro-only baselines against NLP-augmented models:

1. **Baseline:** Linear regression, macroeconomic controls only
2. **Keyword NLP:** Macro + hawkish/dovish keyword scores
3. **FinBERT NLP:** Macro + FinBERT sentiment scores
4. **Full NLP:** Macro + all NLP features (Ridge / Lasso for regularization)
5. **XGBoost:** Macro + all NLP features, nonlinear relationships

Evaluation uses **walk-forward time-series cross-validation** to prevent any look-ahead bias.

### Evaluation Metrics
- **Regression:** RMSE, MAE
- **Classification (direction):** Accuracy, F1-score
- All NLP models compared against macro-only baseline to isolate the value added by text features

---

## Expected Outcomes

We expect FOMC language to contain useful information about short-term Treasury yield movements. In particular, we believe more hawkish or dovish language may help explain how parts of the yield curve react after policy communication. We also expect shorter-term yields (especially the 2-year) to respond more strongly than longer-term yields because they are more sensitive to policy expectations.

A strong result would show that Fed language improves prediction beyond standard macroeconomic and market variables. Even weak results would be meaningful — it may suggest that most information from FOMC communication is already priced in quickly by the market.

As a possible extension, we could study whether statements, minutes, and press conferences have different effects on different parts of the yield curve.

---

## Project Structure

```
ds-project/
├── 00_setup_folders.py         # Run once — creates all data/ and outputs/ subfolders
├── 01_collect_fomc_text.py     # Scrape FOMC statements & minutes from Fed website
├── 02_nlp_pipeline.py          # *** THE CORE *** — NLP feature extraction
├── 03_collect_yield_data.py    # Download Treasury yields + macro data from FRED
├── 04_modeling.py              # Train & evaluate predictive models
├── FOMC_NLP_Colab.ipynb        # Shared Colab notebook (GitHub + Drive workflow)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                       # Gitignored — lives on Google Drive
│   ├── raw/
│   │   ├── fomc_text/          # fomc_statements.json, fomc_minutes.json
│   │   └── fred/               # raw FRED yield and macro CSVs
│   ├── processed/              # fomc_nlp_features.csv, fomc_yield_targets.csv
│   └── splits/                 # X_train, X_test, y_train, y_test CSVs per run
│
└── outputs/                    # Gitignored — lives on Google Drive
    ├── models/                 # Trained model .pkl files (one per model per target)
    ├── predictions/            # Predicted vs actual CSVs (one per model per target)
    ├── results/                # model_results.csv — RMSE / MAE summary table
    ├── feature_importance/     # Feature importance CSVs (one per model per target)
    ├── figures/
    │   ├── nlp/                # Hawkishness timeline, FinBERT score distributions
    │   ├── yields/             # Yield time series, curve shape charts
    │   ├── models/             # RMSE comparison bars, actual-vs-predicted, residuals
    │   └── features/           # Feature importance bars, correlation heatmaps
    └── logs/                   # Run logs and timing
```

---

## Collaboration Setup

We use **GitHub + Google Colab + Google Drive**:
- Code lives in this GitHub repo (`ds-project`)
- Data and outputs persist on shared Google Drive (not committed to git)
- Both partners open `FOMC_NLP_Colab.ipynb` in Colab, which pulls the latest code from GitHub and symlinks `data/` to Drive each session

See `FOMC_NLP_Colab.ipynb` for the full session workflow.

---

## Setup & Installation

```bash
git clone https://github.com/YOUR_USERNAME/ds-project.git
cd ds-project
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# For GPU (strongly recommended for FinBERT):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Run Order

### Step 1 — Collect FOMC Text

**Option A — HuggingFace dataset (fastest, recommended for getting started):**
```bash
# Edit 01_collect_fomc_text.py and uncomment the load_from_huggingface() block
python 01_collect_fomc_text.py
```

**Option B — Scrape Fed website directly (most complete, ~20 min):**
```bash
python 01_collect_fomc_text.py
```

---

### Step 2 — NLP Pipeline (start here — everything depends on this)

```bash
# Keyword-only first (fast, no GPU needed — good for testing):
python 02_nlp_pipeline.py --statements data/fomc_statements.json \
                          --no-finbert --no-embeddings

# Full pipeline with FinBERT (GPU recommended, ~15-30 min):
python 02_nlp_pipeline.py --statements data/fomc_statements.json \
                          --minutes   data/fomc_minutes.json \
                          --device cuda

# CPU-only fallback:
python 02_nlp_pipeline.py --no-finbert --device cpu
```

Output: `data/fomc_nlp_features.csv`

**NLP features produced:**

| Feature | Method | Description |
|---------|--------|-------------|
| `hawk_score_norm` | Keyword lexicon | Hawkish − dovish word frequency per 100 words |
| `hawk_ratio` | Keyword lexicon | Fraction of tone words that are hawkish (0=pure dove, 1=pure hawk) |
| `hawk_score_delta` | Keyword lexicon | Change in hawkishness vs. prior meeting |
| `finbert_net` | FinBERT | Positive − Negative sentiment (chunked, averaged across document) |
| `emb_hawk_net` | Sentence transformer | Embedding similarity to hawkish anchor minus dovish anchor |
| `hawk_composite` | Ensemble | Normalized average of all hawk signals |
| `tfidf_cos_prev` | TF-IDF | Cosine similarity to prior meeting (language drift) |
| `tfidf_pc1/2` | TF-IDF + LSA | Principal vocabulary directions across the full corpus |

---

### Step 3 — Treasury Yield Data

```bash
python 03_collect_yield_data.py

# With a free FRED API key (higher rate limits):
# Get one at https://fred.stlouisfed.org/docs/api/api_key.html
python 03_collect_yield_data.py --api-key YOUR_KEY_HERE
```

Output: `data/fred_yields.csv`, `data/fomc_yield_targets.csv`

**Target variables:**
- `yield_2y_chg1d`, `yield_2y_chg3d`, `yield_2y_chg5d` — 2-year yield changes
- `yield_10y_chg1d`, `yield_10y_chg3d`, `yield_10y_chg5d` — 10-year yield changes
- `spread_2s10s_chg*` — yield curve slope changes (2s10s spread)
- `*_dir*d` — binary direction (1 = yield rose, 0 = fell) for classification tasks

---

### Step 4 — Modeling

```bash
python 04_modeling.py --doc-type statement   # Statements only
python 04_modeling.py --doc-type minutes     # Minutes only
python 04_modeling.py --doc-type both        # All documents combined
```

Output: `outputs/model_results.csv`, `outputs/feature_importance_*.csv`

---

## Key Design Decisions

**Why keyword scores before FinBERT?**
Keyword scores are fast, interpretable, and surprisingly competitive. FinBERT adds marginal lift but requires GPU and significant runtime. Build the keyword baseline first, then add FinBERT incrementally.

**Why chunk FinBERT?**
FOMC statements are 400–600 words; minutes are 5,000–10,000 words. FinBERT's max context is 512 tokens. We split documents into ~400-word chunks, score each, and average. This preserves signal from the full document rather than truncating.

**Why merge_asof for date matching?**
FOMC meeting dates in raw text (e.g., "January 28-29, 2020") don't always match FRED's daily dates exactly. `merge_asof` with a 30-day tolerance handles this robustly without manual alignment.

**Why the 2-year yield as primary target?**
The 2-year yield is most sensitive to near-term monetary policy expectations, so FOMC language should have the strongest predictive signal there. We compare against 10-year and the 2s10s spread to study transmission along the full curve.

---

## Potential Pitfalls

- **Look-ahead bias:** Always use walk-forward CV. The current TF-IDF is fit on the full corpus — this is acceptable for research but should be noted in the writeup.
- **Market timing:** FOMC statements are released at 2pm ET; minutes ~3 weeks later. The "1-day" change captures the overnight + next-day reaction.
- **Regime changes:** The 2022–2023 hiking cycle looks very different from 2010–2019. Consider subsample analyses or regime indicator variables.
- **Efficient markets:** Weak results are still publishable — they may indicate FOMC language is priced in near-instantly.
