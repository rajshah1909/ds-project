"""
01_collect_fomc_text.py
=======================
Scrapes FOMC policy statements and meeting minutes from the Federal Reserve
website and saves them as structured JSON files.

Outputs:
  data/fomc_statements.json  — one record per meeting (2000–present)
  data/fomc_minutes.json     — one record per meeting (2005–present)

Usage:
  python 01_collect_fomc_text.py

Notes:
  - Be polite: the script sleeps between requests to avoid hammering the Fed.
  - Run once, then work from the saved JSON files.
  - Alternatively, the HuggingFace dataset `seanchua/FOMC` already contains
    cleaned statements; see the comment at the bottom of this file.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = f"{BASE_URL}/monetarypolicy/fomc_historical.htm"
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "FOMC-Research-Project/1.0 (academic use)"}
SLEEP_SEC = 1.5  # polite crawl delay


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP_SEC)
    return BeautifulSoup(resp.text, "html.parser")


def clean_text(raw: str) -> str:
    """Strip boilerplate, extra whitespace, and formatting artifacts."""
    # Remove vote tallies (e.g., "Voting for the FOMC monetary policy action were:")
    raw = re.sub(r"Voting for the FOMC.*?(?=\n\n|\Z)", "", raw, flags=re.DOTALL)
    # Remove implementation notes section (operational details, not policy language)
    raw = re.sub(
        r"(Implementation Note|Open Market Operations|Desk Operations).*?\Z",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Collapse whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def extract_text_from_url(url: str) -> str:
    """Download a Fed page and extract the main article text."""
    try:
        soup = get_soup(url)
        # Fed pages use <div id="article"> or <div class="col-xs-12 col-sm-8 col-md-8">
        article = (
            soup.find("div", id="article")
            or soup.find("div", class_=re.compile(r"col-xs-12"))
            or soup.find("div", class_="panel-default")
        )
        if article is None:
            article = soup.find("body")
        return clean_text(article.get_text(separator="\n"))
    except Exception as exc:
        print(f"  ⚠ Could not fetch {url}: {exc}")
        return ""


# ── Step 1: Collect links from the historical archive page ───────────────────
def scrape_historical_links() -> list[dict]:
    """
    Parse fomc_historical.htm and return a list of:
      {year, date_str, statement_url, minutes_url}
    for every historical meeting.
    """
    print("Scraping historical archive page …")
    soup = get_soup(HISTORICAL_URL)
    records = []

    for panel in soup.find_all("div", class_="panel"):
        header = panel.find(class_=re.compile(r"panel-heading"))
        if header is None:
            continue
        year_text = header.get_text(strip=True)
        try:
            year = int(re.search(r"\d{4}", year_text).group())
        except (AttributeError, ValueError):
            continue

        for row in panel.find_all("div", class_=re.compile(r"row")):
            links = {a.get_text(strip=True).lower(): a["href"] for a in row.find_all("a", href=True)}
            # Date is usually in a <strong> or first text node of the row
            date_span = row.find("strong") or row.find("em")
            date_str = date_span.get_text(strip=True) if date_span else ""

            stmt_url = next(
                (BASE_URL + v for k, v in links.items() if "statement" in k), None
            )
            min_url = next(
                (BASE_URL + v for k, v in links.items() if "minute" in k), None
            )

            if stmt_url or min_url:
                records.append(
                    {
                        "year": year,
                        "date_str": date_str,
                        "statement_url": stmt_url,
                        "minutes_url": min_url,
                    }
                )

    print(f"  Found {len(records)} historical meeting entries")
    return records


# ── Step 2: Collect links from the current calendar page ────────────────────
def scrape_calendar_links() -> list[dict]:
    """
    Parse fomccalendars.htm to get recent and upcoming meetings.
    Returns same schema as scrape_historical_links().
    """
    print("Scraping calendar page …")
    soup = get_soup(CALENDAR_URL)
    records = []

    for panel in soup.find_all("div", class_="panel"):
        header = panel.find(class_=re.compile(r"panel-heading"))
        if header is None:
            continue
        year_text = header.get_text(strip=True)
        try:
            year = int(re.search(r"\d{4}", year_text).group())
        except (AttributeError, ValueError):
            continue

        for row in panel.find_all("div", class_=re.compile(r"fomc-meeting")):
            links = {a.get_text(strip=True).lower(): a["href"] for a in row.find_all("a", href=True)}
            date_div = row.find(class_=re.compile(r"date"))
            date_str = date_div.get_text(strip=True) if date_div else ""

            stmt_url = next(
                (BASE_URL + v for k, v in links.items() if "statement" in k), None
            )
            min_url = next(
                (BASE_URL + v for k, v in links.items() if "minute" in k), None
            )

            if stmt_url or min_url:
                records.append(
                    {
                        "year": year,
                        "date_str": date_str,
                        "statement_url": stmt_url,
                        "minutes_url": min_url,
                    }
                )

    print(f"  Found {len(records)} calendar entries")
    return records


# ── Step 3: Fetch and save text ──────────────────────────────────────────────
def build_corpus(links: list[dict]) -> tuple[list, list]:
    statements, minutes = [], []

    for i, rec in enumerate(links):
        print(f"  [{i+1}/{len(links)}] {rec['year']} — {rec['date_str']}")

        if rec.get("statement_url"):
            text = extract_text_from_url(rec["statement_url"])
            if text:
                statements.append(
                    {
                        "year": rec["year"],
                        "date_str": rec["date_str"],
                        "url": rec["statement_url"],
                        "text": text,
                        "word_count": len(text.split()),
                    }
                )

        if rec.get("minutes_url"):
            text = extract_text_from_url(rec["minutes_url"])
            if text:
                minutes.append(
                    {
                        "year": rec["year"],
                        "date_str": rec["date_str"],
                        "url": rec["minutes_url"],
                        "text": text,
                        "word_count": len(text.split()),
                    }
                )

    return statements, minutes


# ── Alternative: Use the HuggingFace FOMC dataset ───────────────────────────
def load_from_huggingface() -> list[dict]:
    """
    Faster alternative: load the pre-cleaned FOMC dataset from HuggingFace.
    `pip install datasets`

    The seanchua/FOMC dataset contains statements from 1993–2023 with
    columns: date, text, title, speaker, url.

    Returns records in the same schema used by the rest of this pipeline.
    """
    from datasets import load_dataset

    ds = load_dataset("seanchua/FOMC", split="train")
    records = []
    for row in ds:
        records.append(
            {
                "year": int(row["date"][:4]) if row.get("date") else None,
                "date_str": row.get("date", ""),
                "url": row.get("url", ""),
                "text": clean_text(row.get("text", "")),
                "word_count": len(row.get("text", "").split()),
                "source": "huggingface/seanchua/FOMC",
            }
        )
    print(f"Loaded {len(records)} records from HuggingFace dataset")
    return records


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Option A: Scrape directly from the Fed website (slower, most complete)
    hist_links = scrape_historical_links()
    cal_links = scrape_calendar_links()
    all_links = hist_links + cal_links

    # Deduplicate by URL
    seen = set()
    unique_links = []
    for r in all_links:
        key = (r.get("statement_url"), r.get("minutes_url"))
        if key not in seen:
            seen.add(key)
            unique_links.append(r)

    print(f"\nFetching text for {len(unique_links)} unique meetings …")
    statements, minutes = build_corpus(unique_links)

    # Save
    stmt_path = OUT_DIR / "fomc_statements.json"
    mins_path = OUT_DIR / "fomc_minutes.json"
    stmt_path.write_text(json.dumps(statements, indent=2))
    mins_path.write_text(json.dumps(minutes, indent=2))

    print(f"\n✓ Saved {len(statements)} statements → {stmt_path}")
    print(f"✓ Saved {len(minutes)} minutes     → {mins_path}")

    # Option B (comment out A and uncomment this):
    # hf_records = load_from_huggingface()
    # (OUT_DIR / "fomc_statements_hf.json").write_text(json.dumps(hf_records, indent=2))
