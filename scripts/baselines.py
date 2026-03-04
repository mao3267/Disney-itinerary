import json
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Ride:
    key: str
    name: str
    rating: float            # happiness score
    duration_min: float      # ride duration in minutes
    drive_min: float         # one-way center -> ride in minutes
    wait_mean_min: float     # expected wait time in minutes

    @property
    def cost_min(self) -> float:
        # Must return to center after each ride
        return 2.0 * self.drive_min + self.wait_mean_min + self.duration_min


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rides(
    ride_metadata_path: Path,
    driving_times_path: Path,
    wait_stats_path: Path,
) -> List[Ride]:
    meta = load_json(ride_metadata_path)
    driving = load_json(driving_times_path)
    wait_stats = load_json(wait_stats_path)

    mu: Dict[str, float] = wait_stats.get("mu", {})
    if not isinstance(mu, dict) or len(mu) == 0:
        raise ValueError(f"'mu' not found or empty in {wait_stats_path}")

    rides: List[Ride] = []
    for r in meta["rides"]:
        key = r["dataset_name"]

        if key not in driving:
            raise KeyError(f"Missing driving time for ride '{key}' in {driving_times_path}")
        if key not in mu:
            raise KeyError(f"Missing wait mean (mu) for ride '{key}' in {wait_stats_path}")

        rides.append(
            Ride(
                key=key,
                name=str(r["name"]),
                rating=float(r["rating"]),
                duration_min=float(r["duration_min"]),
                drive_min=float(driving[key]),
                wait_mean_min=float(mu[key]),
            )
        )

    return rides


def summarize_plan(selected: List[Ride]) -> Tuple[float, float]:
    total_rating = sum(r.rating for r in selected)
    total_time = sum(r.cost_min for r in selected)
    return total_rating, total_time


# ----------------------------
# Baseline 1: random feasible selection within budget
# ----------------------------
def baseline_random(rides: List[Ride], budget_min: float, seed: int = 0) -> List[Ride]:
    rng = random.Random(seed)
    remaining = budget_min

    pool = rides[:]
    rng.shuffle(pool)

    chosen: List[Ride] = []
    for r in pool:
        if r.cost_min <= remaining + 1e-9:
            chosen.append(r)
            remaining -= r.cost_min
    return chosen


# ----------------------------
# Bruteforce utilities
# ----------------------------
def _subset_plan_from_mask(rides: List[Ride], mask: int) -> List[Ride]:
    chosen: List[Ride] = []
    for i, r in enumerate(rides):
        if (mask >> i) & 1:
            chosen.append(r)
    return chosen


def _subset_cost_rating(rides: List[Ride], mask: int) -> Tuple[float, float, int]:
    t = 0.0
    s = 0.0
    c = 0
    for i, r in enumerate(rides):
        if (mask >> i) & 1:
            t += r.cost_min
            s += r.rating
            c += 1
    return t, s, c


# ----------------------------
# Baseline 2: maximize total rating (BRUTEFORCE exact)
# ----------------------------
def baseline_max_rating_bruteforce(rides: List[Ride], budget_min: float) -> List[Ride]:
    n = len(rides)
    if n > 25:
        raise ValueError(f"Bruteforce is exponential; n={n} is too large for this baseline.")

    best_mask = 0
    best_rating = -1e100
    best_time = 1e100
    best_count = -1

    for mask in range(1 << n):
        t, s, c = _subset_cost_rating(rides, mask)
        if t > budget_min + 1e-9:
            continue

        # Primary: max rating
        # Tie-break 1: use MORE of the budget (optional; makes plans comparable)
        # Tie-break 2: more rides
        if (s > best_rating + 1e-12) or (
            abs(s - best_rating) <= 1e-12 and t > best_time + 1e-9
        ) or (
            abs(s - best_rating) <= 1e-12 and abs(t - best_time) <= 1e-9 and c > best_count
        ):
            best_rating = s
            best_time = t
            best_count = c
            best_mask = mask

    return _subset_plan_from_mask(rides, best_mask)


# ----------------------------
# Baseline 3: maximize number of rides (BRUTEFORCE exact)
#   - tie-break: max rating
# ----------------------------
def baseline_max_count_bruteforce(rides: List[Ride], budget_min: float) -> List[Ride]:
    n = len(rides)
    if n > 25:
        raise ValueError(f"Bruteforce is exponential; n={n} is too large for this baseline.")

    best_mask = 0
    best_count = -1
    best_rating = -1e100
    best_time = 1e100

    for mask in range(1 << n):
        t, s, c = _subset_cost_rating(rides, mask)
        if t > budget_min + 1e-9:
            continue

        # Primary: max count
        # Tie-break 1: max rating
        # Tie-break 2: use MORE of the budget
        if (c > best_count) or (
            c == best_count and s > best_rating + 1e-12
        ) or (
            c == best_count and abs(s - best_rating) <= 1e-12 and t > best_time + 1e-9
        ):
            best_count = c
            best_rating = s
            best_time = t
            best_mask = mask

    return _subset_plan_from_mask(rides, best_mask)


# ----------------------------
# Baseline 4: greedy by rating
# ----------------------------
def baseline_greedy_rating_desc(rides: List[Ride], budget_min: float) -> List[Ride]:
    pool = sorted(rides, key=lambda r: (-r.rating, r.cost_min, r.key))
    chosen: List[Ride] = []
    remaining = budget_min

    for r in pool:
        if r.cost_min <= remaining + 1e-9:
            chosen.append(r)
            remaining -= r.cost_min

    
    chosen_set = {r.key for r in chosen}
    stable = [r for r in rides if r.key in chosen_set]
    return stable


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_processed = repo_root / "data" / "processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_min", type=float, required=True, help="Park open-time budget in minutes")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["random", "max_rating", "max_count", "greedy_rating"],
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for baseline=random")

    parser.add_argument("--ride_metadata", type=str, default=str(default_processed / "ride_metadata.json"))
    parser.add_argument("--driving_times", type=str, default=str(default_processed / "driving_times.json"))
    parser.add_argument("--wait_stats", type=str, default=str(default_processed / "wait_stats_all.json"))

    args = parser.parse_args()

    rides = build_rides(
        Path(args.ride_metadata),
        Path(args.driving_times),
        Path(args.wait_stats),
    )

    if args.baseline == "random":
        chosen = baseline_random(rides, args.budget_min, seed=args.seed)
    elif args.baseline == "max_rating":
        chosen = baseline_max_rating_bruteforce(rides, args.budget_min)
    elif args.baseline == "max_count":
        chosen = baseline_max_count_bruteforce(rides, args.budget_min)
    else:
        chosen = baseline_greedy_rating_desc(rides, args.budget_min)

    total_rating, total_time = summarize_plan(chosen)

    print(f"Baseline: {args.baseline}")
    print(f"Budget (min): {args.budget_min:.2f}")
    print(f"Total time used (min): {total_time:.2f}")
    print(f"Total rating: {total_rating:.2f}")
    print("Rides picked (in order):")
    for r in chosen:
        print(f"  - {r.key:22s} | {r.name:30s} | rating={r.rating:.1f} | cost={r.cost_min:.2f} min")


if __name__ == "__main__":
    main()