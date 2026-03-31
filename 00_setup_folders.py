"""
00_setup_folders.py
===================
Run once at the start of the project to create all folders.
Safe to re-run — never deletes existing files.

Usage:
    python 00_setup_folders.py
"""

from pathlib import Path

FOLDERS = [
    # ── Raw data ──────────────────────────────────────────────────────────────
    "data/raw/fomc_text",        # raw scraped JSON (statements, minutes)
    "data/raw/fred",             # raw FRED CSVs (yields, macro)

    # ── Processed data ────────────────────────────────────────────────────────
    "data/processed",            # cleaned, merged, feature-engineered CSVs

    # ── Train / test splits ───────────────────────────────────────────────────
    "data/splits",               # X_train, X_test, y_train, y_test CSVs

    # ── Trained models ────────────────────────────────────────────────────────
    "outputs/models",            # saved model files (.pkl, .json)

    # ── Predictions & scores ──────────────────────────────────────────────────
    "outputs/predictions",       # per-meeting predicted vs actual CSVs

    # ── Evaluation results ────────────────────────────────────────────────────
    "outputs/results",           # RMSE/MAE/F1 summary tables

    # ── Feature importance ────────────────────────────────────────────────────
    "outputs/feature_importance",# feature importance CSVs per model

    # ── Charts & figures ──────────────────────────────────────────────────────
    "outputs/figures/nlp",       # hawkishness over time, FinBERT scores, etc.
    "outputs/figures/yields",    # yield time series, curve shapes
    "outputs/figures/models",    # RMSE comparison bars, residual plots
    "outputs/figures/features",  # feature importance bars, correlation heatmaps

    # ── Logs ──────────────────────────────────────────────────────────────────
    "outputs/logs",              # run logs, timing, errors
]

def setup():
    created = []
    for folder in FOLDERS:
        p = Path(folder)
        p.mkdir(parents=True, exist_ok=True)
        gitkeep = p / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
            created.append(str(p))

    if created:
        print("Created folders:")
        for f in created:
            print(f"  ✓ {f}/")
    else:
        print("All folders already exist — nothing to do.")

    print("\nProject layout:")
    for folder in FOLDERS:
        print(f"  {folder}/")

if __name__ == "__main__":
    setup()
