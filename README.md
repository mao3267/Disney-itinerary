# Disney Itinerary Optimizer

**CSE 203B — Convex Optimization**
**UC San Diego, Winter 2026**

A robust Second-Order Cone Program (SOCP) itinerary optimizer for Walt Disney World Florida. Given uncertain ride wait times, the solver selects an optimal subset of rides that maximizes enjoyment while keeping total wait time within a budget — even under worst-case distributional uncertainty.

## Data Preprocessing

### Data Source

Wait time data comes from the [`touringplans`](https://github.com/touringplans/data) R package, which provides timestamped posted and actual wait times for Walt Disney World attractions. We selected **14 rides** across **4 parks**:

| Park | Rides |
|------|-------|
| Magic Kingdom | Pirates of the Caribbean, Seven Dwarfs Mine Train, Splash Mountain |
| Epcot | Soarin' Around the World, Spaceship Earth |
| Hollywood Studios | Alien Swirling Saucers, Rock 'n' Roller Coaster, Slinky Dog Dash, Toy Story Mania! |
| Animal Kingdom | DINOSAUR, Expedition Everest, Avatar Flight of Passage, Kilimanjaro Safaris, Na'vi River Journey |

Ride ratings (1–10 scale) are computed as an equal-weighted average of four sources, each normalized to 1–10: TouringPlans expert ratings, TouringPlans user reviews, TripAdvisor bubble ratings, and Mousehacking editorial rankings. Range: 5.8 (Alien Saucers) to 9.6 (Flight of Passage). See [`data/rating_methodology.md`](data/rating_methodology.md) for raw scores, normalization formulas, and per-ride calculations.

### Cleaning Steps

| Step | Setting | Rationale |
|------|---------|-----------|
| Wait metric | `wait_minutes_actual` | Reflects real guest experience; posted times inflate both mean and variance |
| Outlier filter | Keep [0, 300] min | Removes erroneous values (e.g., -92918, 90387) |
| COVID exclusion | Mar 2020 — Dec 2020 | Anomalous operational patterns |
| Date window | 2018-06-30 to 2021-08-31 | Common window where all 14 rides have data |

### Aggregation Pipeline

The pipeline (`scripts/preprocess.py`) transforms raw timestamped observations into per-season mean vectors and covariance matrices:

```
Raw CSV (per ride)
  → Hourly aggregation   (group by park_date + hour, mean actual wait)
  → Daily aggregation    (mean of hourly averages, require >= 3 hourly obs/day)
  → Wide format          (one row per date, 14 columns for rides)
  → Season labels        (left-join parks metadata, map wdwseason → 3 categories)
  → Compute μ and Σ      (per season)
  → Save .npz + .json
```

### Season Classification

The `wdwseason` column from TouringPlans park metadata (96% non-null in `parks_metadata.csv`; 77% of daily_means rows receive a label after left-join) is mapped to three categories:

| Category | wdwseason Values |
|----------|-----------------|
| **Peak** | Christmas, Christmas Peak, Easter, July 4th, Summer Break, Thanksgiving, Memorial Day, Martin Luther King Junior Day, Mardi Gras |
| **Regular** | Spring, Presidents Week, Jersey Week, Columbus Day, Halloween |
| **Value** | Fall, September Low, Winter |

### Covariance Estimation

With 14 rides but sparse complete-case data (as few as 0% to 6% of days per season have all 14 rides reporting actual waits), covariance estimation requires regularization:

- **Ledoit-Wolf shrinkage** — Used when complete-case count >= 15 (i.e., n >= p+1). Automatically shrinks the sample covariance toward a structured target, producing a well-conditioned estimate. Applied to the "all seasons" split (n=38, shrinkage=0.394).
- **Pairwise covariance with shrinkage** — Used when complete cases are too few. Computes covariance from all available pairwise observations, then applies 50% shrinkage toward the diagonal and ensures positive semi-definiteness via eigenvalue correction. Applied to peak (n=4), regular (n=11), and value (n=0) splits.

All covariance matrices are verified PSD via Cholesky decomposition before saving.

### Output Files

```
data/processed/
  wait_stats_{season}.npz    # μ (14,) and Σ (14,14) for CVXPY solver
  wait_stats_{season}.json   # Human-readable version with metadata
  ride_metadata.json         # Names, parks, durations, ratings
  daily_means.csv            # 836 rows x 16 cols (date index + 14 rides + season label)
```

Where `{season}` is one of: `all`, `peak`, `regular`, `value`.

## Key Results

### Mean Wait Times (μ, in minutes)

| Season | Min μ | Max μ | Complete-Case Days | Total Days | Method |
|--------|-------|-------|--------------------|------------|--------|
| All | 8.97 (Spaceship Earth) | 52.35 (Flight of Passage) | 38 | 836 | Ledoit-Wolf (shrinkage=0.394) |
| Peak | 8.17 (Spaceship Earth) | 76.40 (Flight of Passage) | 4 | 246 | Pairwise (shrinkage=0.5) |
| Regular | 9.12 (Spaceship Earth) | 65.54 (Flight of Passage) | 11 | 176 | Pairwise (shrinkage=0.5) |
| Value | 7.82 (Spaceship Earth) | 57.94 (Flight of Passage) | 0 | 219 | Pairwise (shrinkage=0.5) |

Flight of Passage consistently has the highest mean wait. Spaceship Earth has the lowest. Peak season inflates top-ride waits by ~32% over value season.

### Condition Numbers (min eigenvalue of Σ)

| Season | Min Eigenvalue |
|--------|----------------|
| All | 31.39 |
| Peak | 22.45 |
| Regular | 25.97 |
| Value | 26.69 |

All matrices are well-conditioned for the SOCP solver.

## How to Run

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- R with the `touringplans` package (only needed for data export)

### Setup

```bash
# Install Python dependencies
uv sync

# (Optional) Export raw data from R — only needed if data/raw/ is empty
Rscript scripts/export_data.R
```

### Run the Preprocessing Pipeline

```bash
uv run python scripts/preprocess.py
```

This produces all files in `data/processed/`.

### Run the SOCP Solver

```bash
uv run python scripts/touristplan.py
```

## Project Structure

```
Disney-itinerary/
├── scripts/
│   ├── export_data.R       # R script to export raw CSVs from touringplans package
│   ├── explore.py          # EDA / data exploration
│   ├── preprocess.py       # Stage 2: aggregation pipeline (μ, Σ)
│   └── touristplan.py      # Stage 3: Robust SOCP solver (CVXPY)
├── data/
│   ├── raw/                # 14 ride CSVs + parks_metadata.csv
│   └── processed/          # μ/Σ stats, ride_metadata, daily_means
├── report/                 # LaTeX report
├── plan.md                 # Project roadmap
├── stage2.md               # Stage 2 design doc & results
└── pyproject.toml          # Python project config
```
