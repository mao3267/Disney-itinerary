import json
import random
import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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
    

def load_wait_stats_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    mu = np.asarray(data["mu"], dtype=float)
    sigma = np.asarray(data["sigma"], dtype=float)
    ride_names = [str(x) for x in data["ride_names"].tolist()]
    return mu, sigma, ride_names


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

    best_mask = 0
    best_rating = -1e100
    best_time = -1e100
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

    best_mask = 0
    best_count = -1
    best_rating = -1e100
    best_time = -1e100

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


# ----------------------------
# simulate plan with variance
# ----------------------------
def simulate_plan_time_once(
    selected: List[Ride],
    mu_vec: np.ndarray,
    sigma_mat: np.ndarray,
    ride_names: List[str],
    rng: np.random.Generator,
    clip_min: float = 0.0,
    clip_max: float = 300.0,
) -> float:
    name_to_idx = {name: i for i, name in enumerate(ride_names)}
    idx = [name_to_idx[r.key] for r in selected]

    sub_mu = mu_vec[idx]
    sub_sigma = sigma_mat[np.ix_(idx, idx)]

    sampled_waits = rng.multivariate_normal(mean=sub_mu, cov=sub_sigma)
    sampled_waits = np.clip(sampled_waits, clip_min, clip_max)

    total_time = 0.0
    for ride, sampled_wait in zip(selected, sampled_waits):
        total_time += 2.0 * ride.drive_min + ride.duration_min + float(sampled_wait)

    return total_time


def simulate_overrun_probability(
    selected: List[Ride],
    budget_min: float,
    wait_stats_npz_path: Path,
    n_sim: int = 1000,
    seed: int = 0,
    clip_min: float = 0.0,
    clip_max: float = 300.0,
) -> dict:
    mu_vec, sigma_mat, ride_names = load_wait_stats_npz(wait_stats_npz_path)
    rng = np.random.default_rng(seed)

    simulated_total_times: List[float] = []
    overruns: List[float] = []

    for _ in range(n_sim):
        total_time = simulate_plan_time_once(
            selected=selected,
            mu_vec=mu_vec,
            sigma_mat=sigma_mat,
            ride_names=ride_names,
            rng=rng,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        simulated_total_times.append(total_time)
        overruns.append(max(0.0, total_time - budget_min))

    over_count = sum(t > budget_min + 1e-9 for t in simulated_total_times)
    prob_over = over_count / n_sim

    sorted_times = sorted(simulated_total_times)
    q95 = sorted_times[int(0.95 * (n_sim - 1))]
    q99 = sorted_times[int(0.99 * (n_sim - 1))]

    positive_overruns = [x for x in overruns if x > 0]
    avg_overrun_if_over = (
        sum(positive_overruns) / len(positive_overruns)
        if positive_overruns else 0.0
    )

    return {
        "n_sim": n_sim,
        "prob_overrun": prob_over,
        "mean_total_time": float(sum(simulated_total_times) / n_sim),
        "std_total_time": float(statistics.pstdev(simulated_total_times)),
        "p95_total_time": float(q95),
        "p99_total_time": float(q99),
        "avg_overrun_if_over": float(avg_overrun_if_over),
        "max_total_time": float(max(simulated_total_times)),
        "min_total_time": float(min(simulated_total_times)),
    }


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

    parser.add_argument("--n_sim", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument(
        "--simulate_overrun",
        action="store_true",
        help="Run Monte Carlo simulation to estimate probability of exceeding budget"
    )
    parser.add_argument(
        "--wait_stats_npz",
        type=str,
        default=str(default_processed / "wait_stats_all.npz"),
        help="Path to .npz file containing full mu and sigma"
    )

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

        total_rating, total_time = summarize_plan(chosen)

    print(f"Baseline: {args.baseline}")
    print(f"Budget (min): {args.budget_min:.2f}")
    print(f"Deterministic total time used (using mean waits): {total_time:.2f}")
    print(f"Total rating: {total_rating:.2f}")
    print("Rides picked (in order):")
    for r in chosen:
        print(f"  - {r.key:22s} | {r.name:30s} | rating={r.rating:.1f} | cost={r.cost_min:.2f} min")

    if args.simulate_overrun:
        sim = simulate_overrun_probability(
            selected=chosen,
            budget_min=args.budget_min,
            wait_stats_npz_path=Path(args.wait_stats_npz),
            n_sim=args.n_sim,
            seed=args.seed,
        )

        print("\nMonte Carlo overrun experiment")
        print(f"  Simulations: {sim['n_sim']}")
        print(f"  P(total_time > budget): {sim['prob_overrun']:.4f}")
        print(f"  Mean simulated total time: {sim['mean_total_time']:.2f} min")
        print(f"  Std simulated total time: {sim['std_total_time']:.2f} min")
        print(f"  95th percentile total time: {sim['p95_total_time']:.2f} min")
        print(f"  99th percentile total time: {sim['p99_total_time']:.2f} min")
        print(f"  Avg overrun | overrun happened: {sim['avg_overrun_if_over']:.2f} min")
        print(f"  Min / Max simulated total time: {sim['min_total_time']:.2f} / {sim['max_total_time']:.2f} min")


if __name__ == "__main__":
    main()