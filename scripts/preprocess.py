"""Stage 2: Data aggregation pipeline.

Produces μ (mean vector) and Σ (covariance matrix) per season
for the Robust SOCP solver.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RIDES = [
    "alien_saucers",
    "dinosaur",
    "expedition_everest",
    "flight_of_passage",
    "kilimanjaro_safaris",
    "navi_river",
    "pirates_of_caribbean",
    "rock_n_rollercoaster",
    "seven_dwarfs_train",
    "slinky_dog",
    "soarin",
    "spaceship_earth",
    "splash_mountain",
    "toy_story_mania",
]

# Season mapping: wdwseason (uppercase) -> 3-category
SEASON_MAP = {
    # Peak
    "CHRISTMAS": "peak",
    "CHRISTMAS PEAK": "peak",
    "EASTER": "peak",
    "JULY 4TH": "peak",
    "SUMMER BREAK": "peak",
    "THANKSGIVING": "peak",
    "MEMORIAL DAY": "peak",
    "MARTIN LUTHER KING JUNIOR DAY": "peak",
    "MARDI GRAS": "peak",
    # Regular
    "SPRING": "regular",
    "PRESIDENTS WEEK": "regular",
    "JERSEY WEEK": "regular",
    "COLUMBUS DAY": "regular",
    "HALLOWEEN": "regular",
    # Value
    "FALL": "value",
    "SEPTEMBER LOW": "value",
    "WINTER": "value",
}

DATE_START = pd.Timestamp("2018-06-30")
DATE_END = pd.Timestamp("2021-08-31")
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-12-31")

WAIT_MIN = 0
WAIT_MAX = 300
MIN_HOURLY_OBS_PER_DAY = 3


def load_and_clean(ride_name: str) -> pd.DataFrame:
    """Load a ride CSV and apply cleaning filters."""
    df = pd.read_csv(
        RAW_DIR / f"{ride_name}.csv",
        parse_dates=["park_date", "wait_datetime"],
    )

    # Restrict to common date window
    df = df[(df["park_date"] >= DATE_START) & (df["park_date"] <= DATE_END)]

    # Exclude COVID period
    df = df[~((df["park_date"] >= COVID_START) & (df["park_date"] <= COVID_END))]

    # Outlier filter on actual waits
    mask = df["wait_minutes_actual"].notna()
    df.loc[mask & ((df["wait_minutes_actual"] < WAIT_MIN) |
                   (df["wait_minutes_actual"] > WAIT_MAX)),
           "wait_minutes_actual"] = np.nan

    # Extract hour for hourly aggregation
    df["hour"] = df["wait_datetime"].dt.hour

    return df


def hourly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by park_date + hour, compute mean actual wait."""
    hourly = (
        df.groupby(["park_date", "hour"], observed=True)
        .agg(wait_actual_avg=("wait_minutes_actual", "mean"))
        .reset_index()
    )
    # Drop rows where actual mean is NaN (no actual observations that hour)
    hourly = hourly.dropna(subset=["wait_actual_avg"])
    return hourly


def daily_aggregate(hourly: pd.DataFrame) -> pd.DataFrame:
    """Compute daily mean from hourly averages, requiring min observations."""
    daily = (
        hourly.groupby("park_date", observed=True)
        .agg(
            daily_mean=("wait_actual_avg", "mean"),
            hourly_count=("wait_actual_avg", "count"),
        )
        .reset_index()
    )
    # Require minimum hourly observations per day
    daily = daily[daily["hourly_count"] >= MIN_HOURLY_OBS_PER_DAY]
    return daily[["park_date", "daily_mean"]]


def build_wide_table() -> pd.DataFrame:
    """Build wide-format table: rows=dates, columns=rides' daily means."""
    all_daily = {}

    for ride in RIDES:
        print(f"  Processing {ride}...")
        df = load_and_clean(ride)
        hourly = hourly_aggregate(df)
        daily = daily_aggregate(hourly)
        daily = daily.rename(columns={"daily_mean": ride})
        all_daily[ride] = daily.set_index("park_date")[ride]

    wide = pd.DataFrame(all_daily)
    wide.index.name = "park_date"
    return wide


def add_season_labels(wide: pd.DataFrame) -> pd.DataFrame:
    """Join parks metadata to get season labels."""
    meta = pd.read_csv(RAW_DIR / "parks_metadata.csv", parse_dates=["date"])
    meta = meta[["date", "wdwseason"]].rename(columns={"date": "park_date"})
    meta["season"] = meta["wdwseason"].map(SEASON_MAP)

    wide = wide.reset_index()
    wide = wide.merge(meta[["park_date", "season"]], on="park_date", how="left")
    wide = wide.set_index("park_date")
    return wide


def compute_stats(
    wide: pd.DataFrame, season_filter: str
) -> dict:
    """Compute μ and Σ for a given season filter.

    Returns dict with mu, sigma, n_days, method, ride_names.
    """
    if season_filter == "all":
        subset = wide
    else:
        subset = wide[wide["season"] == season_filter]

    ride_cols = [c for c in subset.columns if c != "season"]

    # Complete-case analysis: rows where ALL 14 rides have data
    complete = subset[ride_cols].dropna()
    n_complete = len(complete)

    print(f"  Season={season_filter}: {len(subset)} total days, "
          f"{n_complete} complete-case days")

    n_rides = len(ride_cols)

    if n_complete >= 5 * n_rides:
        # Sufficient data: use Ledoit-Wolf shrinkage (always better than naive)
        mu = complete.mean().values.copy()
        lw = LedoitWolf().fit(complete.values)
        sigma = lw.covariance_.copy()
        method = "ledoit_wolf"
        shrinkage = lw.shrinkage_
        print(f"    Ledoit-Wolf shrinkage coefficient: {shrinkage:.4f}")
    elif n_complete >= n_rides + 1:
        # Borderline: Ledoit-Wolf shrinkage is essential
        mu = complete.mean().values.copy()
        lw = LedoitWolf().fit(complete.values)
        sigma = lw.covariance_.copy()
        method = "ledoit_wolf_borderline"
        shrinkage = lw.shrinkage_
        print(f"    Ledoit-Wolf (borderline) shrinkage: {shrinkage:.4f}")
    else:
        # Too few complete cases: use pairwise covariance with shrinkage
        print(f"    WARNING: Only {n_complete} complete cases. "
              f"Using pairwise covariance with shrinkage.")
        data = subset[ride_cols]
        mu = data.mean().values.copy()
        # Pairwise covariance: use all available pairs
        sigma = data.cov().values.copy()
        # Handle NaN in covariance (can happen with very sparse data)
        nan_mask = np.isnan(sigma)
        if nan_mask.any():
            sigma[nan_mask] = 0.0
        # Apply shrinkage toward diagonal to regularize pairwise estimate
        diag = np.diag(np.diag(sigma))
        shrinkage = 0.5
        sigma = (1 - shrinkage) * sigma + shrinkage * diag
        # Regularize if still not PSD
        min_eig = np.linalg.eigvalsh(sigma).min()
        if min_eig < 0:
            sigma += (-min_eig + 1e-6) * np.eye(n_rides)
        method = "pairwise_shrunk"

    # Validate mu has no NaN
    if np.isnan(mu).any():
        nan_rides = [ride_cols[i] for i in range(len(mu)) if np.isnan(mu[i])]
        raise ValueError(
            f"NaN in mean vector for season={season_filter}. "
            f"Rides with no data: {nan_rides}"
        )

    # Ensure PSD with progressive regularization
    for eps in [1e-8, 1e-6, 1e-4, 1e-2]:
        eigenvalues = np.linalg.eigvalsh(sigma)
        if eigenvalues.min() >= 1e-10:
            break
        sigma += eps * np.eye(n_rides)
        print(f"    Added eps={eps:.0e} for PSD correction")

    eigenvalues = np.linalg.eigvalsh(sigma)
    print(f"    Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")

    # Verify Cholesky works
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Cannot make Σ PSD for season={season_filter} "
            f"after regularization: {e}"
        )

    return {
        "mu": mu,
        "sigma": sigma,
        "n_days": n_complete,
        "n_total_days": len(subset),
        "method": method,
        "shrinkage": float(shrinkage) if shrinkage is not None else None,
        "ride_names": ride_cols,
        "min_eigenvalue": float(eigenvalues.min()),
    }


def save_stats(stats: dict, season_filter: str) -> None:
    """Save μ and Σ to .npz and .json files."""
    prefix = f"wait_stats_{season_filter}"

    # .npz for CVXPY solver
    np.savez(
        PROCESSED_DIR / f"{prefix}.npz",
        mu=stats["mu"],
        sigma=stats["sigma"],
        ride_names=stats["ride_names"],
        n_days=stats["n_days"],
    )

    # .json for human inspection
    json_data = {
        "season": season_filter,
        "n_days_complete": stats["n_days"],
        "n_days_total": stats["n_total_days"],
        "method": stats["method"],
        "shrinkage": stats["shrinkage"],
        "min_eigenvalue": stats["min_eigenvalue"],
        "ride_names": stats["ride_names"],
        "mu": {
            name: round(float(val), 2)
            for name, val in zip(stats["ride_names"], stats["mu"])
        },
        "sigma_diagonal": {
            name: round(float(stats["sigma"][i, i]), 2)
            for i, name in enumerate(stats["ride_names"])
        },
    }
    with open(PROCESSED_DIR / f"{prefix}.json", "w") as f:
        json.dump(json_data, f, indent=2)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 2: Data Aggregation Pipeline")
    print("=" * 60)

    # Build wide-format daily means table
    print("\n[1/4] Building wide-format daily means...")
    wide = build_wide_table()

    # Add season labels
    print("\n[2/4] Adding season labels...")
    wide = add_season_labels(wide)

    # Save daily means CSV
    print("\n[3/4] Saving daily_means.csv...")
    wide.to_csv(PROCESSED_DIR / "daily_means.csv")
    print(f"  Shape: {wide.shape}")

    # Report season distribution
    season_counts = wide["season"].value_counts(dropna=False)
    print(f"\n  Season distribution:")
    for season, count in season_counts.items():
        label = season if pd.notna(season) else "unlabeled"
        print(f"    {label}: {count}")

    # Compute and save stats for each season
    print("\n[4/4] Computing μ and Σ...")
    seasons = ["all", "peak", "regular", "value"]
    for season in seasons:
        print(f"\n  --- {season.upper()} ---")
        stats = compute_stats(wide, season)
        save_stats(stats, season)
        print(f"    μ range: [{stats['mu'].min():.1f}, {stats['mu'].max():.1f}]")
        print(f"    Saved {season} stats")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for season in seasons:
        data = np.load(PROCESSED_DIR / f"wait_stats_{season}.npz")
        print(f"\n  {season}: μ shape={data['mu'].shape}, "
              f"Σ shape={data['sigma'].shape}, "
              f"n_days={int(data['n_days'])}")
    print(f"\n  Output directory: {PROCESSED_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
