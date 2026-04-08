#!/usr/bin/env python3
"""
Build data_centers_geocoded.csv for the Streamlit topics map (white AI hub markers).

The app does not geocode at page load: ~1,300 ArcGIS calls × rate limit ≈ 15–25+ minutes
and the browser looks frozen. Run this script once when data_centers_global.csv changes.

Usage (from this folder):
  python geocode_data_centers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import ArcGIS

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
INPUT_CSV = DATA_DIR / "data_centers_global.csv"
OUTPUT_CSV = DATA_DIR / "data_centers_geocoded.csv"


def main() -> None:
    if not INPUT_CSV.is_file():
        print(f"Missing {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    dc_df = (
        pd.read_csv(INPUT_CSV)
        .groupby(["City", "Country"], as_index=False)
        .agg({"Total_Data_Centers": "sum"})
    )
    n = len(dc_df)
    geolocator = ArcGIS(timeout=15)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=0.12)

    rows: list[dict] = []
    for i, row in dc_df.iterrows():
        city, country = row["City"], row["Country"]
        q = f"{city}, {country}"
        try:
            loc = geocode(q)
            if loc:
                rows.append(
                    {
                        "City": city,
                        "Country": country,
                        "Total_Data_Centers": int(row["Total_Data_Centers"]),
                        "latitude": loc.latitude,
                        "longitude": loc.longitude,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            print(f"skip {q!r}: {exc}", file=sys.stderr)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  {i + 1} / {n} …", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows with coordinates).")


if __name__ == "__main__":
    main()
