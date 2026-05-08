# Decoding the Fed: Using NLP to Predict Yield Curve Movements

## Project Overview

This project looks at whether Federal Reserve FOMC language can help predict short-term movements in the U.S. Treasury yield curve.

The Federal Reserve releases statements after FOMC meetings. These statements are important because investors use them to understand the Fed's view on inflation, unemployment, interest rates, and the economy. Even when the Fed does not change interest rates, the wording of the statement can still affect market expectations.

In this project, we use Natural Language Processing (NLP) to measure whether FOMC statements sound more hawkish or dovish. Then we test whether those language features help predict changes in Treasury yields after each meeting.

## Main Question

Can FOMC language help predict short-term Treasury yield movements?

More specifically, we look at:

- 2-year Treasury yield changes
- 10-year Treasury yield changes
- 2s10s yield spread changes
- 1-day, 3-day, and 5-day changes after FOMC meetings

## Authors

| Name | NetID |
|------|-------|
| Charles Timmes | cwt32 |
| Raj Shah | ras637 |

## GitHub Repository

```text
https://github.com/rajshah1909/ds-project
```

## How This Project Was Run

The final project was mainly run in Google Colab using:

```text
FOMC_NLP_Colab.ipynb
```

The Python files are also included in the repo as script versions of the same workflow. They are useful for showing the full project structure and for running the project locally if needed.

## Repository Structure

```text
ds-project/
│
├── 00_setup_folders.py
├── 01_collect_fomc_text.py
├── 02_nlp_pipeline.py
├── 03_collect_yield_data.py
├── 04_modeling.py
├── FOMC_NLP_Colab.ipynb
├── README.md
├── SCHEMA.md
├── requirements.txt
│
└── fomc_nlp/
    ├── data/
    └── outputs/
```

## Folder Details

### `fomc_nlp/data/`

This folder contains the project data files.

```text
fomc_nlp/data/
├── fomc_minutes.json
├── fomc_statements.json
├── fomc_nlp_keyword.csv
├── fomc_nlp_features.csv
├── fred_yields.csv
└── fomc_yield_targets.csv
```

### `fomc_nlp/outputs/`

This folder contains the saved model outputs.

```text
fomc_nlp/outputs/
├── feature_importance/
├── figures/
├── models/
├── predictions/
└── results/
```

## Main Files

### `FOMC_NLP_Colab.ipynb`

This is the main notebook used for the project.

It does the full workflow:

1. Mounts Google Drive
2. Clones the GitHub repo
3. Installs packages
4. Collects FOMC statements
5. Creates NLP features
6. Downloads yield data
7. Builds target variables
8. Runs regression models
9. Runs classification models
10. Saves results and charts

### `00_setup_folders.py`

Creates the folders used by the project.

This script sets up the project folder structure before running the rest of the pipeline.

### `01_collect_fomc_text.py`

Collects FOMC statements and minutes from the Federal Reserve website.

The main output from this step is FOMC text data saved in JSON format.

### `02_nlp_pipeline.py`

Creates NLP features from FOMC text.

It creates features such as:

- hawkish word count
- dovish word count
- net hawkishness score
- hawkishness per 100 words
- TF-IDF similarity to the previous meeting
- FinBERT sentiment scores

### `03_collect_yield_data.py`

Downloads Treasury yield and macro data.

The data includes:

- 2-year Treasury yield
- 5-year Treasury yield
- 10-year Treasury yield
- federal funds rate
- CPI
- unemployment rate

It also creates the target variables for 1-day, 3-day, and 5-day yield changes.

### `04_modeling.py`

Runs the machine learning models and saves the results.

Models used include:

- Macro-only baseline
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- XGBoost
- Logistic Regression for direction prediction
- Random Forest Classifier for direction prediction

## Data Sources

### FOMC Text

FOMC statements were collected from the Federal Reserve website.

```text
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
```

### Treasury Yield and Macro Data

Treasury yield and macroeconomic data came from FRED.

Main FRED series used:

```text
DGS2      = 2-Year Treasury Constant Maturity Rate
DGS5      = 5-Year Treasury Constant Maturity Rate
DGS10     = 10-Year Treasury Constant Maturity Rate
DFF       = Federal Funds Effective Rate
CPIAUCSL  = Consumer Price Index
UNRATE    = Unemployment Rate
```

### NLP Model

FinBERT was used for financial sentiment scoring.

```text
ProsusAI/finbert
```

## Data Files

### `fomc_statements.json`

Stores FOMC statement text.

Main fields:

```text
year
fomc_date
url
text
word_count
```

### `fomc_minutes.json`

Stores FOMC minutes text when available.

### `fomc_nlp_keyword.csv`

Stores keyword-based NLP features.

Main columns:

```text
fomc_date
year
word_count
hawk_hits
dove_hits
hawk_net
hawk_score_norm
hawk_ratio
tone_word_count
tfidf_cos_prev
tfidf_pc1
tfidf_pc2
hawk_score_delta
hawk_ratio_delta
```

Column meaning:

```text
hawk_hits
- number of hawkish terms found in the statement

dove_hits
- number of dovish terms found in the statement

hawk_net
- hawkish hits minus dovish hits

hawk_score_norm
- hawkish score normalized per 100 words

hawk_ratio
- share of tone words that are hawkish

tone_word_count
- total hawkish and dovish terms found

tfidf_cos_prev
- text similarity compared to the previous FOMC statement

tfidf_pc1
- first TF-IDF / LSA component

tfidf_pc2
- second TF-IDF / LSA component

hawk_score_delta
- change in hawkishness from the previous meeting

hawk_ratio_delta
- change in hawkish ratio from the previous meeting
```

### `fomc_nlp_features.csv`

Stores the full NLP feature table.

This includes keyword features and FinBERT sentiment features.

Main columns include:

```text
hawk_score_norm
hawk_ratio
hawk_net
tone_word_count
tfidf_cos_prev
tfidf_pc1
tfidf_pc2
finbert_positive
finbert_negative
finbert_neutral
finbert_net
hawk_composite
```

### `fred_yields.csv`

Stores Treasury yield and macro data.

Main columns:

```text
yield_2y
yield_5y
yield_10y
fed_funds_rate
cpi
unemployment
cpi_yoy
spread_2s10s
spread_5s10s
```

Column meaning:

```text
yield_2y
- 2-year Treasury yield

yield_5y
- 5-year Treasury yield

yield_10y
- 10-year Treasury yield

fed_funds_rate
- federal funds effective rate

cpi
- consumer price index

unemployment
- unemployment rate

cpi_yoy
- year-over-year CPI change

spread_2s10s
- 10-year yield minus 2-year yield

spread_5s10s
- 10-year yield minus 5-year yield
```

### `fomc_yield_targets.csv`

Stores the prediction targets.

Regression targets include:

```text
yield_2y_chg1d
yield_2y_chg3d
yield_2y_chg5d

yield_10y_chg1d
yield_10y_chg3d
yield_10y_chg5d

spread_2s10s_chg1d
spread_2s10s_chg3d
spread_2s10s_chg5d
```

Direction targets include:

```text
yield_2y_dir1d
yield_2y_dir3d
yield_2y_dir5d

yield_10y_dir1d
yield_10y_dir3d
yield_10y_dir5d

spread_2s10s_dir1d
spread_2s10s_dir3d
spread_2s10s_dir5d
```

Column meaning:

```text
chg1d
- yield change after 1 business day

chg3d
- yield change after 3 business days

chg5d
- yield change after 5 business days

dir1d
- 1 if the yield moved up after 1 day, otherwise 0

dir3d
- 1 if the yield moved up after 3 days, otherwise 0

dir5d
- 1 if the yield moved up after 5 days, otherwise 0
```

## Method

The project follows this workflow:

```text
Collect FOMC text
        ↓
Clean and process text
        ↓
Create NLP features
        ↓
Download FRED yield data
        ↓
Create yield movement targets
        ↓
Train models
        ↓
Compare results
        ↓
Save charts and output files
```

## NLP Features

The project uses three main types of NLP features.

### 1. Hawkish and Dovish Keyword Scores

We created lists of hawkish and dovish words.

Examples of hawkish words:

```text
tightening
restrictive
rate hike
elevated inflation
price pressures
further increases
```

Examples of dovish words:

```text
easing
accommodative
rate cut
downside risk
slowing
pause
```

These words were counted in each FOMC statement.

### 2. TF-IDF Features

TF-IDF was used to compare each FOMC statement to the previous statement.

This helps measure how much the Fed changed its language from one meeting to the next.

### 3. FinBERT Sentiment

FinBERT was used to score financial sentiment.

Each statement was split into chunks, scored with FinBERT, and then averaged into one document-level score.

## Models

The project uses regression models to predict the size of yield changes.

Regression models:

```text
Linear Regression
Ridge Regression
Lasso Regression
Random Forest
XGBoost
```

The project also uses classification models to predict whether yields move up or down.

Classification models:

```text
Logistic Regression
Random Forest Classifier
```

## Evaluation

Regression models were evaluated using:

```text
RMSE
MAE
```

Classification models were evaluated using:

```text
Accuracy
F1 Score
Majority Baseline
```

Walk-forward time-series validation was used so that older meetings were used to predict later meetings. This avoids look-ahead bias.

## Main Results

The strongest result was for predicting the 2-year Treasury yield 1-day change.

Best model:

```text
Random Forest
```

Best result:

```text
RMSE = 0.0865
MAE = 0.0674
```

Macro-only baseline:

```text
RMSE = 0.1098
MAE = 0.0876
```

This means the NLP features improved prediction compared to the macro-only baseline.

## Classification Results

Direction prediction was harder.

For the 2-year yield 1-day direction task:

```text
Majority baseline accuracy = 0.598
Macro-only accuracy = 0.529
Logistic full NLP accuracy = 0.516
Random Forest classifier accuracy = 0.523
```

This means the models did not consistently beat the majority baseline for direction prediction.

## Main Finding

NLP features helped more for predicting the size of yield movements than predicting the direction.

The 2-year yield showed the clearest result because it is more sensitive to near-term Federal Reserve policy expectations.

Keyword-based hawkish and dovish features were useful because FOMC language has special meaning. A statement can sound neutral in normal sentiment but still be hawkish in monetary policy terms.

## Output Files

### Results

Saved in:

```text
fomc_nlp/outputs/results/
```

Important files:

```text
classification_results_final.csv
*_model_results.csv
```

### Predictions

Saved in:

```text
fomc_nlp/outputs/predictions/
```

These files store actual vs predicted values.

### Figures

Saved in:

```text
fomc_nlp/outputs/figures/
```

Important figures:

```text
hawkishness_timeline.png
model_comparison.png
model_comparison_reg_clf.png
rmse_all_targets.png
classification_all_targets.png
feature_importance_rf.png
feature_importance_rf_final.png
```

### Models

Saved in:

```text
fomc_nlp/outputs/models/
```

These are saved `.pkl` model files.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the local scripts in this order:

```bash
python 00_setup_folders.py
python 01_collect_fomc_text.py
python 02_nlp_pipeline.py
python 03_collect_yield_data.py
python 04_modeling.py
```

Or run the full project in Google Colab:

```text
FOMC_NLP_Colab.ipynb
```

The Colab notebook was the main workflow used for the final project.

## Requirements

Main packages used:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
transformers
torch
sentence-transformers
beautifulsoup4
requests
tqdm
```

## Notes

- The `fomc_nlp` folder contains the real data and output folders.
- The project was mainly run in Google Colab.
- The Python scripts are kept as local versions of the same project steps.
- `.DS_Store` files should not be uploaded.
- If model `.pkl` files are too large for GitHub, they can be removed or handled with Git LFS.

## Final Repo Structure

The final repo should look like this:

```text
ds-project/
├── README.md
├── SCHEMA.md
├── requirements.txt
├── FOMC_NLP_Colab.ipynb
├── 00_setup_folders.py
├── 01_collect_fomc_text.py
├── 02_nlp_pipeline.py
├── 03_collect_yield_data.py
├── 04_modeling.py
└── fomc_nlp/
    ├── data/
    └── outputs/
```

## Short Summary

This project uses FOMC statement language to test whether Federal Reserve communication helps predict Treasury yield movements.

The main result is that NLP features improved regression performance for the 2-year Treasury yield 1-day change. Random Forest performed best for this target. Direction prediction was harder and did not consistently beat the majority baseline.
