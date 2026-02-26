"""Stage 1: Exploratory analysis of touringplans data."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

RAW_DIR = Path("data/raw")

RIDES = [
    # Magic Kingdom
    "pirates_of_caribbean", "seven_dwarfs_train", "splash_mountain",
    # Epcot
    "soarin", "spaceship_earth",
    # Hollywood Studios
    "alien_saucers", "rock_n_rollercoaster", "slinky_dog", "toy_story_mania",
    # Animal Kingdom
    "dinosaur", "expedition_everest", "flight_of_passage",
    "kilimanjaro_safaris", "navi_river",
]


def analyze_ride(ride_name: str) -> dict:
    """Analyze a single ride dataset."""
    df = pd.read_csv(
        RAW_DIR / f"{ride_name}.csv",
        parse_dates=["park_date", "wait_datetime"],
    )

    total_rows = len(df)

    actual_notna = df["wait_minutes_actual"].notna().sum()
    posted_notna = df["wait_minutes_posted"].notna().sum()
    actual_pct = actual_notna / total_rows * 100
    posted_pct = posted_notna / total_rows * 100

    date_min = df["park_date"].min()
    date_max = df["park_date"].max()
    unique_dates = df["park_date"].nunique()

    # Dates with at least 1 actual observation
    dates_with_actual = df.loc[df["wait_minutes_actual"].notna(), "park_date"].nunique()
    dates_with_posted = df.loc[df["wait_minutes_posted"].notna(), "park_date"].nunique()

    # Stats for actual
    actual_vals = df["wait_minutes_actual"].dropna()
    actual_stats = {
        "mean": round(actual_vals.mean(), 1),
        "std": round(actual_vals.std(), 1),
        "median": round(actual_vals.median(), 1),
        "min": round(actual_vals.min(), 1),
        "max": round(actual_vals.max(), 1),
    } if len(actual_vals) > 0 else {}

    # Stats for posted
    posted_vals = df["wait_minutes_posted"].dropna()
    posted_stats = {
        "mean": round(posted_vals.mean(), 1),
        "std": round(posted_vals.std(), 1),
        "median": round(posted_vals.median(), 1),
        "min": round(posted_vals.min(), 1),
        "max": round(posted_vals.max(), 1),
    } if len(posted_vals) > 0 else {}

    return {
        "ride": ride_name,
        "total_rows": total_rows,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "unique_dates": unique_dates,
        "actual_notna": int(actual_notna),
        "actual_pct": round(actual_pct, 2),
        "posted_notna": int(posted_notna),
        "posted_pct": round(posted_pct, 2),
        "dates_with_actual": dates_with_actual,
        "dates_with_posted": dates_with_posted,
        "actual_stats": actual_stats,
        "posted_stats": posted_stats,
    }


def analyze_parks_metadata() -> dict:
    """Analyze parks metadata."""
    pm = pd.read_csv(RAW_DIR / "parks_metadata.csv", parse_dates=["date"])

    total_rows = len(pm)
    date_min = pm["date"].min()
    date_max = pm["date"].max()

    # wdw_ticket_season coverage
    ticket_notna = pm["wdw_ticket_season"].notna().sum()
    ticket_pct = ticket_notna / total_rows * 100

    ticket_distribution = {}
    if ticket_notna > 0:
        dist = pm["wdw_ticket_season"].value_counts()
        ticket_distribution = {k: int(v) for k, v in dist.items()}

    # wdwseason coverage
    wdw_notna = pm["wdwseason"].notna().sum()
    wdw_pct = wdw_notna / total_rows * 100

    wdw_distribution = {}
    if wdw_notna > 0:
        dist = pm["wdwseason"].value_counts()
        wdw_distribution = {k: int(v) for k, v in dist.items()}

    return {
        "total_rows": total_rows,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "wdw_ticket_season_coverage": int(ticket_notna),
        "wdw_ticket_season_pct": round(ticket_pct, 2),
        "wdw_ticket_season_distribution": ticket_distribution,
        "wdwseason_coverage": int(wdw_notna),
        "wdwseason_pct": round(wdw_pct, 2),
        "wdwseason_distribution": wdw_distribution,
        "columns": list(pm.columns),
    }


def analyze_attractions_metadata() -> dict:
    """Analyze attractions metadata."""
    am = pd.read_csv(RAW_DIR / "attractions_metadata.csv")
    records = am.to_dict(orient="records")
    return {
        "total_rows": len(am),
        "columns": list(am.columns),
        "data": records,
    }


def main():
    print("=" * 70)
    print("TOURINGPLANS DATA EXPLORATION")
    print("=" * 70)

    # 1. Ride analysis
    print("\n## RIDE DATASETS\n")
    ride_results = []
    for ride_name in RIDES:
        result = analyze_ride(ride_name)
        ride_results.append(result)

    # Summary table
    print(f"{'Ride':<25} {'Rows':>8} {'Date Range':<25} "
          f"{'Actual%':>8} {'Posted%':>8} {'ActDates':>8} {'PostDates':>9}")
    print("-" * 100)
    for r in ride_results:
        print(f"{r['ride']:<25} {r['total_rows']:>8,} "
              f"{r['date_min']} - {r['date_max']} "
              f"{r['actual_pct']:>7.2f}% {r['posted_pct']:>7.2f}% "
              f"{r['dates_with_actual']:>8} {r['dates_with_posted']:>9}")

    # Stats table
    print(f"\n{'Ride':<25} {'Posted Mean':>11} {'Posted Std':>11} "
          f"{'Actual Mean':>11} {'Actual Std':>11}")
    print("-" * 80)
    for r in ride_results:
        pm = r["posted_stats"].get("mean", "N/A")
        ps = r["posted_stats"].get("std", "N/A")
        am = r["actual_stats"].get("mean", "N/A")
        as_ = r["actual_stats"].get("std", "N/A")
        print(f"{r['ride']:<25} {pm:>11} {ps:>11} {am:>11} {as_:>11}")

    # 2. Parks metadata
    print("\n\n## PARKS METADATA\n")
    pm_result = analyze_parks_metadata()
    print(f"Rows: {pm_result['total_rows']}")
    print(f"Date range: {pm_result['date_min']} - {pm_result['date_max']}")
    print(f"Columns: {pm_result['columns']}")
    print(f"\nwdw_ticket_season coverage: {pm_result['wdw_ticket_season_coverage']}/{pm_result['total_rows']} "
          f"({pm_result['wdw_ticket_season_pct']}%)")
    print("Distribution:")
    for season, count in sorted(pm_result["wdw_ticket_season_distribution"].items()):
        print(f"  {season}: {count}")
    print(f"\nwdwseason coverage: {pm_result['wdwseason_coverage']}/{pm_result['total_rows']} "
          f"({pm_result['wdwseason_pct']}%)")
    print("Distribution:")
    for season, count in sorted(pm_result["wdwseason_distribution"].items()):
        print(f"  {season}: {count}")

    # 3. Attractions metadata
    print("\n\n## ATTRACTIONS METADATA\n")
    am_result = analyze_attractions_metadata()
    print(f"Rows: {am_result['total_rows']}")
    print(f"Columns: {am_result['columns']}")
    print(f"\n{'Name':<30} {'Park':<20} {'Duration':>8} {'Wait/100':>8}")
    print("-" * 70)
    for row in am_result["data"]:
        print(f"{row.get('short_name', 'N/A'):<30} {row.get('park', 'N/A'):<20} "
              f"{row.get('duration', 'N/A'):>8} {row.get('average_wait_per_hundred', 'N/A'):>8}")

    # 4. Cross-ride date analysis
    print("\n\n## CROSS-RIDE DATE ANALYSIS\n")
    date_ranges = {r["ride"]: (r["date_min"], r["date_max"]) for r in ride_results}
    all_mins = [r["date_min"] for r in ride_results]
    all_maxs = [r["date_max"] for r in ride_results]
    common_start = max(all_mins)
    common_end = min(all_maxs)
    print(f"Common date window (all 14 rides): {common_start} - {common_end}")
    print(f"\nRides with latest start dates:")
    for r in sorted(ride_results, key=lambda x: x["date_min"], reverse=True)[:5]:
        print(f"  {r['ride']}: starts {r['date_min']}")

    # Save results as JSON for later use
    all_results = {
        "rides": ride_results,
        "parks_metadata": pm_result,
        "attractions_metadata": am_result,
        "common_date_window": {"start": common_start, "end": common_end},
    }
    with open("data/raw/exploration_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n\nResults saved to data/raw/exploration_results.json")


if __name__ == "__main__":
    main()
