import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ENDPOINTS = {
    2024: "https://data.cityofnewyork.us/resource/bqxa-629f.json",
    2025: "https://data.cityofnewyork.us/resource/nt46-nv3c.json",
}
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

RAW_DIR = Path("data/raw")
DAILY_DIR = RAW_DIR / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

# Demand is counted by when the ride was requested (same as the notebook).
TIME_COLUMN = "request_datetime"
PROVIDER_FILTER = "hvfhs_license_num IN ('HV0003', 'HV0005')"  # Uber, Lyft

PAGE_SIZE = 5000   # larger values were rejected by this dataset (HTTP 400)
TIMEOUT = 120      # seconds per request
MAX_WORKERS = 3    # parallel downloads. Set to 1 if you see many 429 / connection errors.

HEADERS = {}
if os.environ.get("SODA_APP_TOKEN"):
    HEADERS["X-App-Token"] = os.environ["SODA_APP_TOKEN"]

session = requests.Session()
session.mount(
    "https://",
    HTTPAdapter(
        pool_maxsize=MAX_WORKERS * 2,
        max_retries=Retry(
            total=4,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        ),
    ),
)

FMT = "%Y-%m-%dT%H:%M:%S"


def fetch_day(day: pd.Timestamp) -> pd.DataFrame:
    """Hour x zone demand counts for one calendar day."""
    start = day.strftime(FMT)
    end = (day + pd.Timedelta(days=1)).strftime(FMT)

    base_params = {
        # This dataset has no hour-truncation function (date_trunc_hh does not exist),
        # so we extract the hour of day (0-23) and rebuild the timestamp below.
        "$select": f"date_extract_hh({TIME_COLUMN}) AS hod, pulocationid, count(*) AS demand",
        "$where": (
            f"{TIME_COLUMN} >= '{start}' AND {TIME_COLUMN} < '{end}' "
            f"AND {PROVIDER_FILTER} AND pulocationid IS NOT NULL"
        ),
        "$group": "hod, pulocationid",
        "$order": "hod, pulocationid",  # stable order is required for offset paging
        "$limit": PAGE_SIZE,
    }

    pages, offset = [], 0
    while True:
        params = {**base_params, "$offset": offset}
        response = session.get(
            ENDPOINTS[day.year], headers=HEADERS, params=params, timeout=TIMEOUT
        )
        if not response.ok:
            print(f"HTTP {response.status_code} for {day.date()}: {response.text[:500]}")
            response.raise_for_status()

        batch = response.json()
        if not batch:
            break
        pages.append(pd.DataFrame(batch))
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not pages:
        return pd.DataFrame(columns=["hour", "pulocationid", "demand"])

    df = pd.concat(pages, ignore_index=True)
    df["hour"] = day + pd.to_timedelta(pd.to_numeric(df["hod"]), unit="h")
    return df[["hour", "pulocationid", "demand"]]


def download_day(day: pd.Timestamp) -> str:
    out = DAILY_DIR / f"hour_zone_{day:%Y-%m-%d}.csv"
    df = fetch_day(day)
    if df.empty:
        raise ValueError("no rows returned")
    tmp = out.with_suffix(".part")  # write to temp first so a crash never leaves a half file
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    return f"{len(df):,} rows"


def download_zone_lookup() -> None:
    out = RAW_DIR / "taxi_zone_lookup.csv"
    if out.exists():
        return
    try:
        response = requests.get(ZONE_LOOKUP_URL, timeout=60)
        response.raise_for_status()
        out.write_bytes(response.content)
        print("Saved", out)
    except Exception as e:
        print(f"Could not download zone lookup ({type(e).__name__}: {e}).")
        print("Download it manually from the TLC page and save it as", out)


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else START_DATE
    end = sys.argv[2] if len(sys.argv) > 2 else END_DATE

    all_days = pd.date_range(start, end, freq="D")
    pending = [d for d in all_days if not (DAILY_DIR / f"hour_zone_{d:%Y-%m-%d}.csv").exists()]
    print(f"{len(all_days)} days requested, {len(all_days) - len(pending)} already done, "
          f"{len(pending)} to download.")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_day, d): d for d in pending}
        for i, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                print(f"[{i}/{len(pending)}] {day.date()}: {future.result()}")
            except Exception as e:
                print(f"[{i}/{len(pending)}] FAILED {day.date()}: {type(e).__name__}: {e}")
                failed.append(day.date().isoformat())

    download_zone_lookup()

    if failed:
        print(f"\n{len(failed)} day(s) failed. Re-run the script to retry them.")
        sys.exit(1)
    print("\nAll days downloaded.")


if __name__ == "__main__":
    main()