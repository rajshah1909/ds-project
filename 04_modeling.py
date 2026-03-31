"""
04_modeling.py
==============
Trains and evaluates predictive models for Treasury yield curve movements
using FOMC NLP features + macroeconomic controls.

All outputs are organised into subfolders under outputs/:

  outputs/models/               saved model .pkl files
  outputs/predictions/          predicted vs actual CSVs per model
  outputs/results/              RMSE / MAE / F1 summary tables
  outputs/feature_importance/   feature importance CSVs per model
  outputs/figures/models/       bar charts, residual plots, actual-vs-predicted
  outputs/figures/features/     feature importance bars, correlation heatmaps

Model progression:
  1. Baseline:    Linear regression, macro controls only
  2. Keyword NLP: Macro + keyword hawkishness scores
  3. FinBERT NLP: Macro + FinBERT sentiment scores
  4. Ridge/Lasso: Full NLP + macro (regularised)
  5. XGBoost:     Full NLP + macro, nonlinear

Evaluation:
  - RMSE, MAE for regression
  - Accuracy, F1 for direction classification
  - Walk-forward time-series cross-validation (NO look-ahead bias)

Usage:
  python 04_modeling.py --doc-type statement
  python 04_modeling.py --doc-type minutes
  python 04_modeling.py --doc-type both
"""

import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Output directories ────────────────────────────────────────────────────────
DIRS = {
    "models":      Path("outputs/models"),
    "predictions": Path("outputs/predictions"),
    "results":     Path("outputs/results"),
    "importance":  Path("outputs/feature_importance"),
    "fig_models":  Path("outputs/figures/models"),
    "fig_feats":   Path("outputs/figures/features"),
    "logs":        Path("outputs/logs"),
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
PALETTE = {
    "baseline": "#888780",
    "keyword":  "#185FA5",
    "finbert":  "#1D9E75",
    "ridge":    "#BA7517",
    "lasso":    "#993556",
    "rf":       "#D85A30",
    "xgb":      "#A32D2D",
}

# ── Feature groups ────────────────────────────────────────────────────────────
MACRO_FEATURES = [
    "fed_funds_rate", "cpi_yoy", "unemployment",
    "yield_2y_pre", "yield_10y_pre", "spread_2s10s_pre",
]
NLP_KEYWORD = [
    "hawk_score_norm", "hawk_ratio", "hawk_net",
    "tone_word_count", "hawk_score_delta", "hawk_ratio_delta",
    "tfidf_cos_prev", "tfidf_pc1", "tfidf_pc2",
]
NLP_FINBERT = ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_net"]
NLP_EMBED   = ["emb_hawk_sim", "emb_dove_sim", "emb_hawk_net"]
NLP_ALL     = NLP_KEYWORD + NLP_FINBERT + NLP_EMBED + ["hawk_composite"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_modeling_data(nlp_path: str, targets_path: str,
                       doc_type: str = "statement") -> pd.DataFrame:
    nlp = pd.read_csv(nlp_path)
    targets = pd.read_csv(targets_path)

    if doc_type != "both":
        nlp = nlp[nlp["doc_type"] == doc_type].copy()

    nlp["date_parsed"]   = pd.to_datetime(nlp["date_str"], errors="coerce")
    targets["fomc_date"] = pd.to_datetime(targets["fomc_date"], errors="coerce")

    merged = pd.merge_asof(
        nlp.sort_values("date_parsed"),
        targets.sort_values("fomc_date"),
        left_on="date_parsed", right_on="fomc_date",
        tolerance=pd.Timedelta("30D"), direction="nearest",
    ).sort_values("date_parsed").reset_index(drop=True)

    print(f"Merged dataset: {len(merged)} rows x {len(merged.columns)} columns")
    return merged


def save_splits(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                target: str, tag: str) -> None:
    """Save the final walk-forward train/test split to data/splits/."""
    split_dir = Path("data/splits") / tag
    split_dir.mkdir(parents=True, exist_ok=True)

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]   # last fold = most recent holdout

    X.iloc[train_idx].to_csv(split_dir / f"X_train_{target}.csv", index=False)
    X.iloc[test_idx].to_csv(split_dir  / f"X_test_{target}.csv",  index=False)
    y.iloc[train_idx].to_frame().to_csv(split_dir / f"y_train_{target}.csv", index=False)
    y.iloc[test_idx].to_frame().to_csv(split_dir  / f"y_test_{target}.csv",  index=False)
    print(f"  Splits saved → data/splits/{tag}/  "
          f"(train={len(train_idx)}, test={len(test_idx)})")


# ── Walk-forward cross-validation ─────────────────────────────────────────────
def walk_forward_eval(X: pd.DataFrame, y: pd.Series,
                      model_name: str, model,
                      n_splits: int = 5) -> tuple[dict, pd.DataFrame]:
    tscv = TimeSeriesSplit(n_splits=n_splits,
                           test_size=max(1, len(X) // (n_splits + 1)))
    rmses, maes = [], []
    pred_rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        mask_tr = X_tr.notna().all(axis=1) & y_tr.notna()
        mask_te = X_te.notna().all(axis=1) & y_te.notna()
        if mask_tr.sum() < 10 or mask_te.sum() < 2:
            continue

        model.fit(X_tr[mask_tr], y_tr[mask_tr])
        preds   = model.predict(X_te[mask_te])
        actuals = y_te[mask_te].values

        for a, p, idx in zip(actuals, preds, X_te[mask_te].index):
            pred_rows.append({"fold": fold, "obs_index": idx,
                              "actual": a, "predicted": p})

        rmses.append(np.sqrt(mean_squared_error(actuals, preds)))
        maes.append(mean_absolute_error(actuals, preds))

    metrics = {
        "model":     model_name,
        "n_folds":   len(rmses),
        "rmse_mean": np.mean(rmses) if rmses else np.nan,
        "rmse_std":  np.std(rmses)  if rmses else np.nan,
        "mae_mean":  np.mean(maes)  if maes  else np.nan,
        "mae_std":   np.std(maes)   if maes  else np.nan,
    }
    return metrics, pd.DataFrame(pred_rows)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def get_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    m = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    elif hasattr(m, "coef_"):
        imp = np.abs(m.coef_)
    else:
        return None
    return (pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def model_color(name: str) -> str:
    nl = name.lower()
    if "baseline" in nl or "macro only" in nl: return PALETTE["baseline"]
    if "keyword" in nl:  return PALETTE["keyword"]
    if "finbert" in nl:  return PALETTE["finbert"]
    if "ridge" in nl:    return PALETTE["ridge"]
    if "lasso" in nl:    return PALETTE["lasso"]
    if "forest" in nl:   return PALETTE["rf"]
    if "xgb" in nl:      return PALETTE["xgb"]
    return PALETTE["baseline"]


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_model_comparison(results_df: pd.DataFrame, target: str, tag: str):
    sub = results_df[results_df["target"] == target].dropna(subset=["rmse_mean"])
    if sub.empty:
        return
    sub = sub.sort_values("rmse_mean")
    colors = [model_color(m) for m in sub["model"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.65)))
    bars = ax.barh(sub["model"], sub["rmse_mean"], color=colors,
                   xerr=sub.get("rmse_std"), capsize=3, alpha=0.85, height=0.55)
    ax.set_xlabel("RMSE (percentage points)")
    ax.set_title(f"Model comparison — {target}", fontweight="bold", pad=12)
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, sub.iterrows()):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{row["rmse_mean"]:.4f}', va="center", fontsize=9)
    plt.tight_layout()
    path = DIRS["fig_models"] / f"{tag}__comparison__{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart → {path.name}")


def plot_actual_vs_predicted(preds_df: pd.DataFrame, model_name: str,
                              target: str, tag: str):
    if preds_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(preds_df["actual"], preds_df["predicted"],
               alpha=0.5, s=20, color=model_color(model_name))
    mn = min(preds_df["actual"].min(), preds_df["predicted"].min())
    mx = max(preds_df["actual"].max(), preds_df["predicted"].max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title("Actual vs predicted", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    preds_df = preds_df.copy()
    preds_df["residual"] = preds_df["actual"] - preds_df["predicted"]
    for fold, grp in preds_df.groupby("fold"):
        ax.plot(range(len(grp)), grp["residual"].values,
                marker="o", ms=3, lw=0.7, alpha=0.7, label=f"Fold {fold}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Observation within fold"); ax.set_ylabel("Residual")
    ax.set_title("Residuals by fold", fontweight="bold")
    ax.legend(fontsize=7, ncol=3)

    fig.suptitle(f"{model_name} — {target}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = DIRS["fig_models"] / f"{tag}__avp__{safe_name(model_name)}__{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(imp_df: pd.DataFrame, model_name: str,
                             target: str, tag: str, top_n: int = 20):
    if imp_df is None or imp_df.empty:
        return
    top = imp_df.head(top_n)
    nlp_features = set(NLP_ALL)
    colors = ["#185FA5" if f in nlp_features else "#888780" for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.38)))
    ax.barh(top["feature"], top["importance"], color=colors, alpha=0.85, height=0.6)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features\n{model_name} — {target}",
                 fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color="#185FA5", label="NLP feature"),
                        Patch(color="#888780", label="Macro feature")],
              fontsize=8, loc="lower right")
    plt.tight_layout()
    path = DIRS["fig_feats"] / f"{tag}__importance__{safe_name(model_name)}__{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Importance chart → {path.name}")


def plot_hawkishness_timeline(df: pd.DataFrame, tag: str):
    if "hawk_score_norm" not in df or "date_parsed" not in df:
        return
    sub = df.dropna(subset=["hawk_score_norm","date_parsed"]).sort_values("date_parsed")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax = axes[0]
    ax.plot(sub["date_parsed"], sub["hawk_score_norm"], color="#D85A30", lw=1.5)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Hawkishness score\n(per 100 words)")
    ax.set_title("FOMC Language Hawkishness Over Time", fontweight="bold")
    for s, e, label in [("2004-06-01","2006-06-01","2004–06 hike"),
                         ("2015-12-01","2018-12-01","2015–18 hike"),
                         ("2022-03-01","2023-07-01","2022–23 hike")]:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color="red")
        ax.text(pd.Timestamp(s), ax.get_ylim()[1]*0.9, label,
                fontsize=7, color="darkred", alpha=0.7)

    ax = axes[1]
    vals = sub["hawk_ratio"] - 0.5
    colors = ["#D85A30" if v > 0 else "#185FA5" for v in vals]
    ax.bar(sub["date_parsed"], vals, color=colors, alpha=0.75, width=20)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Hawk ratio − 0.5\n(+= hawkish, −= dovish)")
    ax.set_xlabel("Date")

    plt.tight_layout()
    path = DIRS["fig_models"] / f"{tag}__hawkishness_timeline.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Timeline chart → {path.name}")


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list[str],
                              target: str, tag: str):
    cols = [c for c in feature_cols + [target] if c in df.columns]
    corr = df[cols].corr()[[target]].drop(target).sort_values(target)
    if corr.empty:
        return

    fig, ax = plt.subplots(figsize=(3, max(4, len(corr) * 0.32)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title(f"Feature correlations\nwith {target}", fontweight="bold", pad=10)
    plt.tight_layout()
    path = DIRS["fig_feats"] / f"{tag}__corr_heatmap__{target}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Correlation heatmap → {path.name}")


# ── Build model zoo ────────────────────────────────────────────────────────────
def build_models() -> dict:
    models = {
        "Baseline (macro only)": Pipeline([
            ("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Linear (keyword NLP)": Pipeline([
            ("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Linear (FinBERT NLP)": Pipeline([
            ("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge (full NLP + macro)": Pipeline([
            ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso (full NLP + macro)": Pipeline([
            ("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=5000))]),
        "Random Forest (full NLP + macro)": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42),
    }
    try:
        import xgboost as xgb
        models["XGBoost (full NLP + macro)"] = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="rmse", verbosity=0, random_state=42)
    except ImportError:
        print("  XGBoost not installed — pip install xgboost")
    return models


def feature_set_for(model_name: str, available: dict) -> list[str]:
    ml = model_name.lower()
    if "baseline" in ml or "macro only" in ml:
        return available["macro"]
    if "keyword" in ml:
        return available["macro"] + available["keyword"]
    if "finbert" in ml:
        return available["macro"] + available["finbert"]
    return available["macro"] + available["all_nlp"]


# ── Main ──────────────────────────────────────────────────────────────────────
def run_experiments(df: pd.DataFrame, target: str, tag: str) -> pd.DataFrame:
    print(f"\n{'='*60}\nTarget: {target}\n{'='*60}")

    avail = lambda cols: [c for c in cols if c in df.columns]
    available = {
        "macro":   avail(MACRO_FEATURES),
        "keyword": avail(NLP_KEYWORD),
        "finbert": avail(NLP_FINBERT),
        "all_nlp": avail(NLP_ALL),
    }
    all_feats = available["macro"] + available["all_nlp"]

    y = df[target].copy()
    models = build_models()
    all_results = []

    plot_correlation_heatmap(df, all_feats, target, tag)

    for model_name, model in models.items():
        feat_cols = feature_set_for(model_name, available)
        X = df[feat_cols].copy()
        print(f"  [{model_name}]  features={len(feat_cols)} ...", end="", flush=True)

        metrics, preds_df = walk_forward_eval(X, y, model_name, model)
        metrics["target"] = target
        metrics["n_features"] = len(feat_cols)
        all_results.append(metrics)
        print(f"  RMSE={metrics.get('rmse_mean', float('nan')):.4f}")

        # Save predictions
        preds_df["model"] = model_name
        preds_df["target"] = target
        pred_path = DIRS["predictions"] / f"{tag}__{safe_name(model_name)}__{target}.csv"
        preds_df.to_csv(pred_path, index=False)

        # Actual vs predicted chart
        plot_actual_vs_predicted(preds_df, model_name, target, tag)

        # Save trained model
        mask = X.notna().all(axis=1) & y.notna()
        try:
            model.fit(X[mask], y[mask])
            model_path = DIRS["models"] / f"{tag}__{safe_name(model_name)}__{target}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"    Model saved → {model_path.name}")
        except Exception as e:
            print(f"    Could not save model: {e}")

        # Feature importance
        imp = get_importance(model, feat_cols)
        if imp is not None:
            imp_path = DIRS["importance"] / f"{tag}__{safe_name(model_name)}__{target}.csv"
            imp.to_csv(imp_path, index=False)
            plot_feature_importance(imp, model_name, target, tag)
            print(f"    Top 5: {imp.head(5)['feature'].tolist()}")

    results_df = pd.DataFrame(all_results)
    plot_model_comparison(results_df, target, tag)
    return results_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nlp",      default="data/processed/fomc_nlp_features.csv")
    p.add_argument("--targets",  default="data/processed/fomc_yield_targets.csv")
    p.add_argument("--doc-type", default="statement",
                   choices=["statement", "minutes", "both"])
    p.add_argument("--out",      default="outputs/results/model_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tag = f"{args.doc_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    df = load_modeling_data(args.nlp, args.targets, doc_type=args.doc_type)

    # Save splits for primary target
    primary = "yield_2y_chg1d"
    if primary in df.columns:
        feat_cols = [c for c in MACRO_FEATURES + NLP_ALL if c in df.columns]
        save_splits(df, df[feat_cols], df[primary], primary, tag)

    # Hawkishness timeline
    plot_hawkishness_timeline(df, tag)

    # Run all targets
    target_cols = [c for c in df.columns if (
        "yield_2y_chg" in c or "yield_10y_chg" in c or "spread_2s10s_chg" in c
    )]
    if not target_cols:
        print("No target columns found. Check fomc_yield_targets.csv merge.")
        raise SystemExit(1)

    all_results = []
    for target in target_cols:
        try:
            res = run_experiments(df, target, tag)
            all_results.append(res)
        except Exception as e:
            print(f"  Failed for {target}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"\nAll results saved -> {out_path}")
        print("\n=== FINAL SUMMARY ===")
        print(combined[["target","model","rmse_mean","mae_mean"]].to_string(index=False))
