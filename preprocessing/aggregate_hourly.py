from pathlib import Path

import pandas as pd

DAILY_DIR = Path("data/raw/daily")
OUT_FILE = Path("data/hourly_zone_demand_2024_2025.csv")


def main() -> None:
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")  # 731 days
    files = [DAILY_DIR / f"hour_zone_{d:%Y-%m-%d}.csv" for d in days]
    missing = [f.stem.replace("hour_zone_", "") for f in files if not f.exists()]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} daily file(s), e.g. {', '.join(missing[:5])}\n"
            "Run download_data.py again before aggregating."
        )

    parts = []
    for f in files:
        part = pd.read_csv(f)
        part.columns = part.columns.str.lower().str.strip()
        parts.append(part[["hour", "pulocationid", "demand"]])

    df = pd.concat(parts, ignore_index=True)
    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df["pulocationid"] = pd.to_numeric(df["pulocationid"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna()

    df = df.groupby(["hour", "pulocationid"], as_index=False)["demand"].sum()
    df = df.sort_values(["hour", "pulocationid"]).reset_index(drop=True)

    df.to_csv(OUT_FILE, index=False)
    print(f"Combined {len(files)} daily files")
    print(f"Saved {OUT_FILE}: {len(df):,} rows")
    print("First hour:", df["hour"].min(), "| Last hour:", df["hour"].max())
    print("Total trips:", f"{int(df['demand'].sum()):,}")


if __name__ == "__main__":
    main()