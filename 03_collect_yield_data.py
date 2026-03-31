"""
03_collect_yield_data.py
========================
Downloads Treasury yield data and key macro controls from FRED,
then constructs the target variables: 1-day, 3-day, and 5-day
yield changes around each FOMC meeting.

Outputs:
  data/fred_yields.csv        — daily yield time series
  data/fomc_yield_targets.csv — per-meeting target variables

Usage:
  python 03_collect_yield_data.py [--api-key YOUR_FRED_KEY]
  
  No API key required — FRED allows anonymous access via direct URL.
  An API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
  gives higher rate limits.

FRED Series used:
  DGS2      — 2-Year Treasury Constant Maturity Rate
  DGS5      — 5-Year Treasury Constant Maturity Rate
  DGS10     — 10-Year Treasury Constant Maturity Rate
  DFF       — Federal Funds Effective Rate (overnight)
  CPIAUCSL  — CPI All Urban Consumers (monthly, YoY change as control)
  UNRATE    — Unemployment Rate (monthly, as control)
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = {
    "DGS2": "yield_2y",
    "DGS5": "yield_5y",
    "DGS10": "yield_10y",
    "DFF": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
}
START_DATE = "1999-01-01"  # First FOMC meeting in 1999


# ── FRED download ─────────────────────────────────────────────────────────────
def fetch_fred_series(series_id: str, api_key: str | None = None) -> pd.Series:
    """Download a FRED series and return a date-indexed pandas Series."""
    params = {
        "series_id": series_id,
        "observation_start": START_DATE,
        "file_type": "json",
    }
    if api_key:
        params["api_key"] = api_key
    else:
        # FRED allows anonymous access with a generic key for low-volume use
        params["api_key"] = "abcdefghijklmnopqrstuvwxyz123456"  # placeholder

    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    series = pd.Series(
        {
            pd.Timestamp(o["date"]): float(o["value"])
            for o in obs
            if o["value"] != "."  # FRED uses "." for missing
        },
        name=series_id,
        dtype=float,
    )
    time.sleep(0.5)
    return series


def download_all_series(api_key: str | None = None) -> pd.DataFrame:
    frames = {}
    for series_id, col_name in FRED_SERIES.items():
        print(f"  Downloading {series_id} ({col_name}) …")
        try:
            s = fetch_fred_series(series_id, api_key)
            frames[col_name] = s
            print(f"    {len(s)} observations ({s.index.min().date()} – {s.index.max().date()})")
        except Exception as exc:
            print(f"    ⚠ Failed: {exc}")

    df = pd.DataFrame(frames)
    df.index.name = "date"

    # Forward-fill yields (weekends / holidays): FRED daily yields have gaps
    yield_cols = [c for c in df.columns if c.startswith("yield") or c == "fed_funds_rate"]
    df[yield_cols] = df[yield_cols].ffill()

    # Monthly macro controls: forward-fill to daily frequency
    for col in ["cpi", "unemployment"]:
        if col in df:
            df[col] = df[col].ffill()

    return df


# ── CPI YoY ──────────────────────────────────────────────────────────────────
def add_cpi_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """Replace raw CPI level with YoY percent change (more useful as a control)."""
    if "cpi" not in df:
        return df
    # Approximate: shift 252 trading days ≈ 1 year
    df["cpi_yoy"] = df["cpi"].pct_change(periods=252) * 100
    return df


# ── Yield curve spreads ───────────────────────────────────────────────────────
def add_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Add 2s10s spread (key measure of curve slope) and 5s10s spread."""
    if "yield_2y" in df and "yield_10y" in df:
        df["spread_2s10s"] = df["yield_10y"] - df["yield_2y"]
    if "yield_5y" in df and "yield_10y" in df:
        df["spread_5s10s"] = df["yield_10y"] - df["yield_5y"]
    return df


# ── FOMC date parsing ─────────────────────────────────────────────────────────
def parse_fomc_dates(nlp_csv: str = "data/fomc_nlp_features.csv") -> pd.DatetimeIndex:
    """
    Extract the release dates from our NLP feature file.
    The date_str column contains strings like "January 28-29, 2020".
    We parse the last (release) date.
    """
    import re

    df = pd.read_csv(nlp_csv)
    dates = []
    for s in df["date_str"].dropna().unique():
        # Match patterns: "January 28-29, 2020", "March 15, 2020", etc.
        m = re.search(
            r"(\w+ [\d\-–]+,?\s*\d{4})", str(s)
        )
        if m:
            raw = m.group(1)
            # Take the last date in a range: "28-29" → "29"
            raw = re.sub(r"(\d+)[–\-](\d+)", r"\2", raw)
            try:
                dates.append(pd.to_datetime(raw))
            except Exception:
                pass
    return pd.DatetimeIndex(sorted(set(dates)))


# ── Build per-FOMC-meeting target variables ───────────────────────────────────
def build_meeting_targets(
    yields_df: pd.DataFrame,
    fomc_dates: pd.DatetimeIndex,
    horizons: list[int] = [1, 3, 5],
) -> pd.DataFrame:
    """
    For each FOMC meeting date, compute:
      - Yield level ON the release date (close)
      - Yield change over 1, 3, 5 business days AFTER the release
      - Direction (1 = up, 0 = down) for classification tasks

    We use the day BEFORE the release as the baseline to capture the
    "surprise" component relative to pre-meeting expectations.
    """
    daily_idx = pd.bdate_range(
        start=yields_df.index.min(), end=yields_df.index.max()
    )
    # Reindex to business days, ffill
    yields_bday = yields_df.reindex(daily_idx).ffill()

    rows = []
    yield_cols = [c for c in yields_df.columns if c.startswith("yield") or c.startswith("spread")]

    for meeting_date in fomc_dates:
        # Find the nearest business day at or after the meeting date
        bday_idx_arr = yields_bday.index
        release_idx = bday_idx_arr.searchsorted(meeting_date)
        if release_idx >= len(bday_idx_arr):
            continue

        release_date = bday_idx_arr[release_idx]
        # Baseline: day BEFORE the meeting (pre-meeting pricing)
        if release_idx == 0:
            continue
        baseline_date = bday_idx_arr[release_idx - 1]

        row = {"fomc_date": meeting_date, "release_date": release_date}

        for col in yield_cols:
            baseline_val = yields_bday.loc[baseline_date, col]
            row[f"{col}_pre"] = baseline_val

            for h in horizons:
                future_idx = release_idx + h
                if future_idx >= len(bday_idx_arr):
                    row[f"{col}_chg{h}d"] = float("nan")
                    row[f"{col}_dir{h}d"] = float("nan")
                else:
                    future_val = yields_bday.iloc[future_idx][col]
                    chg = future_val - baseline_val
                    row[f"{col}_chg{h}d"] = chg
                    row[f"{col}_dir{h}d"] = int(chg > 0)

        # Also add macro controls as of meeting date
        for ctrl in ["fed_funds_rate", "cpi_yoy", "unemployment"]:
            if ctrl in yields_bday.columns:
                row[ctrl] = yields_bday.loc[release_date, ctrl] if release_date in yields_bday.index else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=None, help="FRED API key (optional)")
    p.add_argument("--yields-out", default="data/fred_yields.csv")
    p.add_argument("--targets-out", default="data/fomc_yield_targets.csv")
    p.add_argument("--nlp-features", default="data/fomc_nlp_features.csv",
                   help="NLP feature file (to extract FOMC dates from)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Downloading FRED data …")
    yields = download_all_series(api_key=args.api_key)
    yields = add_cpi_yoy(yields)
    yields = add_spreads(yields)

    yields.to_csv(args.yields_out)
    print(f"✓ Yield data saved → {args.yields_out}  ({len(yields)} days)")

    if Path(args.nlp_features).exists():
        print("\nBuilding FOMC meeting target variables …")
        fomc_dates = parse_fomc_dates(args.nlp_features)
        print(f"  Found {len(fomc_dates)} unique FOMC dates")
        targets = build_meeting_targets(yields, fomc_dates)
        targets.to_csv(args.targets_out, index=False)
        print(f"✓ Meeting targets saved → {args.targets_out}  ({len(targets)} meetings)")
        print(targets.describe().to_string())
    else:
        print(f"\n⚠ {args.nlp_features} not found — run 02_nlp_pipeline.py first, then re-run this script to build targets.")
