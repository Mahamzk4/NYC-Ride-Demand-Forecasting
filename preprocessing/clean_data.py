from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/hourly_zone_demand_2024_2025.csv")
ZONE_LOOKUP_FILE = Path("data/raw/taxi_zone_lookup.csv")  # optional, from download_data.py
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2024-01-01 00:00"
END = "2025-12-31 23:00"

# Zone IDs 264 and 265 are "Unknown" in the TLC lookup (no real location).
# True  -> drop them so the matrix has the 263 real zones
# False -> keep them (matrix will have 265 columns)
EXCLUDE_UNKNOWN_ZONES = True
UNKNOWN_ZONE_IDS = {264, 265}

REQUIRED_COLUMNS = ["hour", "pulocationid", "demand"]


def get_valid_zone_ids() -> list[int]:
    """Zone IDs the matrix will contain, in sorted order."""
    if ZONE_LOOKUP_FILE.exists():
        lookup = pd.read_csv(ZONE_LOOKUP_FILE)
        lookup.columns = lookup.columns.str.lower().str.strip()
        ids = set(lookup["locationid"].astype(int))
        print(f"Zone IDs loaded from {ZONE_LOOKUP_FILE} ({len(ids)} total)")
    else:
        ids = set(range(1, 266))
        print("Zone lookup file not found; assuming zone IDs 1-265")

    if EXCLUDE_UNKNOWN_ZONES:
        ids -= UNKNOWN_ZONE_IDS
    return sorted(ids)


def main() -> None:

    print("Loading hourly demand data...")
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.lower().str.strip()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[REQUIRED_COLUMNS].copy()
    original_rows = len(df)
    print("Original rows:", original_rows)


    df["hour"] = pd.to_datetime(df["hour"], errors="coerce")
    df["pulocationid"] = pd.to_numeric(df["pulocationid"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    df = df[
        (df["pulocationid"] > 0)
        & (df["pulocationid"] % 1 == 0)
        & (df["demand"] >= 0)
        & (df["demand"] % 1 == 0)
    ].copy()

    df["pulocationid"] = df["pulocationid"].astype(int)
    df["demand"] = df["demand"].astype(int)

    # Keep only the study period (2024 and 2025)
    df = df[(df["hour"] >= START) & (df["hour"] <= END)].copy()

    
    zone_ids = get_valid_zone_ids()
    rows_before_zone_filter = len(df)
    unknown_rows = df.loc[~df["pulocationid"].isin(zone_ids), "demand"]
    df = df[df["pulocationid"].isin(zone_ids)].copy()

    print(
        f"Dropped {rows_before_zone_filter - len(df)} rows from zones outside the valid set "
        f"({unknown_rows.sum()} trips)"
    )


    df = df.groupby(["hour", "pulocationid"], as_index=False)["demand"].sum()
    df = df.sort_values(["hour", "pulocationid"]).reset_index(drop=True)
    print("Rows after cleaning:", len(df))

    cleaned_file = OUTPUT_DIR / "hourly_zone_demand_clean.csv"
    df.to_csv(cleaned_file, index=False)
    print("Saved:", cleaned_file)

    matrix = df.pivot_table(
        index="hour",
        columns="pulocationid",
        values="demand",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    # Hours that never appeared in the raw data (record BEFORE reindexing)
    full_hours = pd.date_range(start=START, end=END, freq="h", name="hour")
    missing_hours = full_hours.difference(matrix.index)

    pd.DataFrame({"missing_hour": missing_hours}).to_csv(
        OUTPUT_DIR / "missing_hours_report.csv", index=False
    )
    print("Missing entire hours:", len(missing_hours))

    # Reindex to every hour and every zone. A zone with no pickups in an hour
    # (or an hour with no data) becomes 0 instead of silently disappearing.
    zones_never_seen = sorted(set(zone_ids) - set(matrix.columns))
    matrix = matrix.reindex(index=full_hours, columns=zone_ids, fill_value=0).astype(int)

    print("Number of hours:", len(matrix))
    print("Number of zones:", len(matrix.columns))
    if zones_never_seen:
        print(f"Zones with no trips at all in the data ({len(zones_never_seen)}): {zones_never_seen}")

 
    expected_hours = len(full_hours)  # 17,544 for 2024 (leap year) + 2025
    assert len(matrix) == expected_hours, "Matrix does not cover every hour"
    assert matrix.index.is_monotonic_increasing and matrix.index.is_unique
    assert (matrix.values >= 0).all(), "Negative demand found"
    assert not matrix.isna().any().any(), "NaNs in matrix"

    # Raw totals should equal matrix totals (nothing lost in pivot/reindex)
    assert matrix.values.sum() == df["demand"].sum(), "Total demand changed during pivot"

    matrix_file = OUTPUT_DIR / "hourly_zone_matrix_clean.csv"
    matrix.to_csv(matrix_file)
    print("Saved:", matrix_file)

    pd.DataFrame({"zone_id": zone_ids}).to_csv(OUTPUT_DIR / "zone_ids.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "original_rows",
                "cleaned_rows",
                "number_of_hours",
                "number_of_zones",
                "zones_with_no_trips",
                "missing_entire_hours",
                "total_trips",
                "first_hour",
                "last_hour",
                "unknown_zones_excluded",
            ],
            "value": [
                original_rows,
                len(df),
                len(matrix),
                len(matrix.columns),
                len(zones_never_seen),
                len(missing_hours),
                int(matrix.values.sum()),
                str(matrix.index.min()),
                str(matrix.index.max()),
                EXCLUDE_UNKNOWN_ZONES,
            ],
        }
    )
    summary_file = OUTPUT_DIR / "cleaning_summary.csv"
    summary.to_csv(summary_file, index=False)
    print("Saved:", summary_file)

    print("\nData cleaning complete!")


if __name__ == "__main__":
    main()