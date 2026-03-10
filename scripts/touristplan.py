import json
import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ProblemData:
    ride_names: List[str]
    ratings:    np.ndarray   # (n,) enjoyment scores
    det_costs:  np.ndarray   # (n,) = 2*drive + duration  (deterministic minutes)
    mu:         np.ndarray   # (n,) mean wait times (stochastic)
    sigma:      np.ndarray   # (n,n) wait-time covariance
    L:          np.ndarray   # (n,n) Cholesky factor of sigma  (sigma = L L^T)
    n:          int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.ride_names)


def load_problem(processed_dir: Path, season: str) -> ProblemData:
    npz_path = processed_dir / f"wait_stats_{season}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {npz_path}\n"
            "Run `uv run python scripts/preprocess.py` first."
        )

    npz        = np.load(npz_path)
    mu         = npz["mu"].astype(float)
    sigma      = npz["sigma"].astype(float)
    ride_names = [str(s) for s in npz["ride_names"]]

    meta    = json.loads((processed_dir / "ride_metadata.json").read_text())
    driving = json.loads((processed_dir / "driving_times.json").read_text())

    # Build per-ride arrays aligned to the order in ride_names
    ride_lookup = {r["dataset_name"]: r for r in meta["rides"]}
    ratings   = np.array([float(ride_lookup[k]["rating"])      for k in ride_names])
    durations = np.array([float(ride_lookup[k]["duration_min"]) for k in ride_names])
    drives    = np.array([float(driving[k])                     for k in ride_names])

    det_costs = 2.0 * drives + durations   # walk-there + ride + walk-back (no wait)

    # Cholesky decomposition: sigma = L L^T
    # sigma is already PSD (verified in preprocess.py)
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        # Tiny numerical fix: regularise diagonal
        sigma += np.eye(len(mu)) * 1e-8
        L = np.linalg.cholesky(sigma)

    return ProblemData(
        ride_names=ride_names,
        ratings=ratings,
        det_costs=det_costs,
        mu=mu,
        sigma=sigma,
        L=L,
    )


# ---------------------------------------------------------------------------
# Feasibility check (used by enumerate and verify)
# ---------------------------------------------------------------------------

def robust_cost(x_bin: np.ndarray, data: ProblemData, kappa: float) -> float:
    """Total worst-case time for a binary selection vector x_bin."""
    det  = float((data.det_costs + data.mu) @ x_bin)
    risk = float(kappa * np.linalg.norm(data.L.T @ x_bin))
    return det + risk


# ---------------------------------------------------------------------------
# Method 1: CVXPY MISOCP
# ---------------------------------------------------------------------------

def solve_misocp(
    data: ProblemData,
    budget: float,
    kappa: float,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Solve the Mixed-Integer SOCP with CVXPY.

    Returns (x_binary_array, solver_status).  x is None if infeasible/failed.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy is required. Run: uv sync")

    n = data.n
    x = cp.Variable(n, boolean=True)

    # SOCP constraint: (d + mu)^T x + kappa * ||L^T x||_2 <= B
    # Written as two linear + one SOC constraint for clarity:
    #   s = L^T x  (affine),   ||s||_2 <= slack
    det_wait  = (data.det_costs + data.mu) @ x
    risk_term = cp.norm(data.L.T @ x, 2)

    objective   = cp.Maximize(data.ratings @ x)
    constraints = [det_wait + kappa * risk_term <= budget]

    prob = cp.Problem(objective, constraints)

    # Try solvers in order of preference for MISOCP.
    # HiGHS supports MILP; SCIP/ECOS_BB support MISOCP but may not be installed.
    # For n=14 the enumeration fallback is exact and instantaneous.
    solvers_to_try = ["SCIP", "GLPK_MI", "HIGHS", "ECOS_BB"]
    status = "failed"
    for solver in solvers_to_try:
        try:
            prob.solve(solver=solver, verbose=False)
            status = prob.status
            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                break
        except Exception:
            continue

    if x.value is None:
        return None, status

    return (x.value > 0.5).astype(float), status


# ---------------------------------------------------------------------------
# Method 2: CVXPY continuous relaxation
# ---------------------------------------------------------------------------

def solve_relaxed(
    data: ProblemData,
    budget: float,
    kappa: float,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Solve the continuous SOCP relaxation (x in [0,1]).

    Returns (x_continuous_array, solver_status).
    The fractional objective is an upper bound on the MISOCP optimum.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy is required. Run: uv sync")

    n = data.n
    x = cp.Variable(n)

    det_wait  = (data.det_costs + data.mu) @ x
    risk_term = cp.norm(data.L.T @ x, 2)

    objective   = cp.Maximize(data.ratings @ x)
    constraints = [
        x >= 0,
        x <= 1,
        det_wait + kappa * risk_term <= budget,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver="CLARABEL", verbose=False)

    if x.value is None:
        prob.solve(verbose=False)   # fallback to CVXPY default

    return x.value, prob.status


# ---------------------------------------------------------------------------
# Method 3: Exact enumeration (fast for n <= 25)
# ---------------------------------------------------------------------------

def solve_enumerate(
    data: ProblemData,
    budget: float,
    kappa: float,
) -> Tuple[np.ndarray, str]:
    """
    Exact search over all 2^n subsets, checking the robust SOCP constraint.

    Fast for n=14 (16384 subsets).  Returns (x_binary_array, "optimal").
    """
    n = data.n
    if n > 25:
        raise ValueError(f"Enumeration is exponential; n={n} is too large.")

    # Precompute column vectors for vectorised cost computation
    # cost(mask) = (det_costs + mu) @ x  +  kappa * ||L^T x||_2
    # We iterate with numpy for speed but it's only 16384 iterations, fine in pure Python too.

    best_mask   = 0
    best_rating = -1e18
    best_cost   = 1e18
    best_count  = -1

    for mask in range(1 << n):
        x_bin = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        cost  = robust_cost(x_bin, data, kappa)
        if cost > budget + 1e-9:
            continue

        rating = float(data.ratings @ x_bin)
        count  = int(x_bin.sum())

        # Primary: max rating.  Tie-break: more rides, then higher cost (uses budget better).
        if (
            rating > best_rating + 1e-12
            or (abs(rating - best_rating) <= 1e-12 and count > best_count)
            or (abs(rating - best_rating) <= 1e-12 and count == best_count and cost > best_cost)
        ):
            best_rating = rating
            best_cost   = cost
            best_count  = count
            best_mask   = mask

    x_best = np.array([(best_mask >> i) & 1 for i in range(n)], dtype=float)
    return x_best, "optimal"


# ---------------------------------------------------------------------------
# Result formatting & output
# ---------------------------------------------------------------------------

@dataclass
class SOCPResult:
    method:          str
    season:          str
    budget:          float
    kappa:           float
    confidence_pct:  float
    solver_status:   str
    ride_names:      List[str]
    x:               np.ndarray    # binary selection
    ratings:         np.ndarray
    det_costs:       np.ndarray
    mu:              np.ndarray
    # derived
    selected_names:  List[str]     = field(init=False)
    total_rating:    float         = field(init=False)
    total_det_cost:  float         = field(init=False)   # without robustness term
    total_nominal:   float         = field(init=False)   # det + mean wait
    total_robust:    float         = field(init=False)   # det + mean wait + kappa*risk
    n_rides:         int           = field(init=False)

    def __post_init__(self) -> None:
        sel = self.x > 0.5
        self.selected_names = [self.ride_names[i] for i in range(len(sel)) if sel[i]]
        self.total_rating   = float(self.ratings  @ self.x)
        self.total_det_cost = float(self.det_costs @ self.x)
        self.total_nominal  = float((self.det_costs + self.mu) @ self.x)
        self.total_robust   = float(self.total_nominal)   # will be overwritten below
        self.n_rides        = int(self.x.sum())

    def set_robust_cost(self, data: ProblemData) -> None:
        self.total_robust = robust_cost(self.x, data, self.kappa)


def format_result(res: SOCPResult, data: ProblemData) -> str:
    lines = [
        f"Method:              {res.method}",
        f"Season:              {res.season}",
        f"Budget (min):        {res.budget:.2f}",
        f"Kappa:               {res.kappa:.4f}  (confidence = {res.confidence_pct:.1f}%)",
        f"Solver status:       {res.solver_status}",
        f"",
        f"Total rating:        {res.total_rating:.2f}",
        f"Rides selected:      {res.n_rides}",
        f"",
        f"Time breakdown (selected rides):",
        f"  Deterministic cost (2*drive + duration):  {res.total_det_cost:.2f} min",
        f"  Nominal total (+ mean wait):              {res.total_nominal:.2f} min",
        f"  Robust total  (+ kappa * risk term):      {res.total_robust:.2f} min",
        f"  Budget:                                   {res.budget:.2f} min",
        f"  Budget slack (robust):                    {res.budget - res.total_robust:.2f} min",
        f"",
        f"{'Ride':<26}  {'Name':<32}  {'Rating':>6}  {'Det':>6}  {'μ_wait':>7}  {'x':>4}",
        f"{'-'*26}  {'-'*32}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*4}",
    ]
    for i, name in enumerate(res.ride_names):
        ride_info = next(r for r in data.__class__.__mro__)   # placeholder
        sel = "  ✓" if res.x[i] > 0.5 else "   "
        lines.append(
            f"{name:<26}  {name:<32}  {data.ratings[i]:>6.1f}  "
            f"{data.det_costs[i]:>6.2f}  {data.mu[i]:>7.2f}  {sel}"
        )
    return "\n".join(lines)


def format_result_v2(res: SOCPResult, data: ProblemData, ride_display_names: dict) -> str:
    """Clean formatted output matching baselines.py style."""
    lines = [
        f"Method:              {res.method}",
        f"Season:              {res.season}",
        f"Budget (min):        {res.budget:.2f}",
        f"Kappa:               {res.kappa:.4f}  (confidence = {res.confidence_pct:.1f}%)",
        f"Solver status:       {res.solver_status}",
        "",
        f"Total rating:        {res.total_rating:.2f}",
        f"Rides selected:      {res.n_rides}",
        "",
        "Time breakdown:",
        f"  Det. cost  (2*drive + duration):          {res.total_det_cost:.2f} min",
        f"  Nominal    (det + mean wait):             {res.total_nominal:.2f} min",
        f"  Robust     (nominal + κ·‖L^T x‖₂):       {res.total_robust:.2f} min",
        f"  Budget:                                   {res.budget:.2f} min",
        f"  Slack (budget − robust):                  {res.budget - res.total_robust:.2f} min",
        "",
        f"{'Key':<26}  {'Display Name':<34}  {'Rating':>6}  {'Det':>6}  {'μ wait':>7}  {'Sel':>4}",
        f"{'-'*26}  {'-'*34}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*4}",
    ]
    for i, name in enumerate(res.ride_names):
        sel = " ✓" if res.x[i] > 0.5 else "  "
        dname = ride_display_names.get(name, name)
        lines.append(
            f"{name:<26}  {dname:<34}  {data.ratings[i]:>6.1f}  "
            f"{data.det_costs[i]:>6.2f}  {data.mu[i]:>7.2f}  {sel:>4}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kappa sweep
# ---------------------------------------------------------------------------

def run_sweep(
    data: ProblemData,
    budget: float,
    kappas: List[float],
    method: str,
    season: str,
) -> List[dict]:
    rows = []
    for kappa in kappas:
        confidence_pct = NormalDist().cdf(kappa) * 100.0
        x, status = _dispatch_solver(data, budget, kappa, method)
        if x is None:
            x = np.zeros(data.n)
        rows.append({
            "kappa":          kappa,
            "confidence_pct": confidence_pct,
            "total_rating":   float(data.ratings @ x),
            "total_nominal":  float((data.det_costs + data.mu) @ x),
            "total_robust":   robust_cost(x, data, kappa),
            "n_rides":        int(x.sum()),
            "solver_status":  status,
            "rides_selected": "|".join(
                sorted(data.ride_names[i] for i in range(data.n) if x[i] > 0.5)
            ),
        })
    return rows


def _dispatch_solver(
    data: ProblemData,
    budget: float,
    kappa: float,
    method: str,
) -> Tuple[Optional[np.ndarray], str]:
    if method == "relax":
        return solve_relaxed(data, budget, kappa)
    elif method == "enumerate":
        return solve_enumerate(data, budget, kappa)
    else:  # misocp — fall back to enumerate if no MI solver is available
        x, status = solve_misocp(data, budget, kappa)
        if x is None:
            # n=14 → enumeration is exact and O(2^14)=16384 subsets, always fast
            print(
                "  [info] No MISOCP solver available (need SCIP/GLPK_MI/ECOS_BB). "
                "Using exact enumeration (n=14, optimal).",
                file=sys.stderr,
            )
            x, status = solve_enumerate(data, budget, kappa)
        return x, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    repo_root        = Path(__file__).resolve().parents[1]
    default_proc     = repo_root / "data" / "processed"
    default_out      = repo_root / "outputs" / "socp"

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Problem spec
    p.add_argument(
        "--season",
        choices=["all", "peak", "regular", "value"],
        default="all",
        help="Which seasonal wait-time statistics to use (default: all).",
    )
    p.add_argument(
        "--budget_min",
        type=float,
        default=600.0,
        metavar="MINUTES",
        help="Total time budget in minutes (default: 600).",
    )

    # Robustness level — exactly one of kappa or confidence_pct
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--kappa",
        type=float,
        default=None,
        metavar="κ",
        help=(
            "Robustness coefficient κ = Φ⁻¹(confidence). "
            "κ=0 → risk-neutral; κ=1.645 → 95%% confidence. "
            "Mutually exclusive with --confidence_pct."
        ),
    )
    group.add_argument(
        "--confidence_pct",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Desired confidence level as a percentage (e.g. 95 for 95%%). "
            "Converted to κ = Φ⁻¹(pct/100). "
            "Mutually exclusive with --kappa."
        ),
    )

    # Method
    p.add_argument(
        "--method",
        choices=["misocp", "relax", "enumerate"],
        default="enumerate",  # exact & fast for n=14; misocp falls back to this anyway
        help=(
            "misocp    — CVXPY Mixed-Integer SOCP (default).\n"
            "relax     — CVXPY continuous SOCP relaxation (x in [0,1]).\n"
            "enumerate — Exact brute-force over all 2^n subsets."
        ),
    )

    # Sweep mode
    p.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run a sweep over κ ∈ [0, 3] and print a CSV + optionally save a plot. "
            "Ignores --kappa / --confidence_pct."
        ),
    )

    # Paths
    p.add_argument("--processed_dir", type=str, default=str(default_proc))
    p.add_argument("--output_dir",    type=str, default=str(default_out))
    p.add_argument(
        "--no_save",
        action="store_true",
        help="Do not save output files; print to stdout only.",
    )

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir    = Path(args.output_dir)

    # ---- Resolve kappa ----
    if args.kappa is not None:
        kappa          = args.kappa
        confidence_pct = NormalDist().cdf(kappa) * 100.0
    elif args.confidence_pct is not None:
        if not (0.0 < args.confidence_pct < 100.0):
            parser.error("--confidence_pct must be in (0, 100).")
        kappa          = NormalDist().inv_cdf(args.confidence_pct / 100.0)
        confidence_pct = args.confidence_pct
    else:
        # Default: 90% confidence
        confidence_pct = 90.0
        kappa          = NormalDist().inv_cdf(confidence_pct / 100.0)

    # ---- Load data ----
    print(f"Loading data: season={args.season}, processed_dir={processed_dir}")
    data = load_problem(processed_dir, args.season)
    print(f"  {data.n} rides loaded.")

    # Load display names for pretty-printing
    meta = json.loads((processed_dir / "ride_metadata.json").read_text())
    ride_display = {r["dataset_name"]: r["name"] for r in meta["rides"]}

    # ---- Sweep mode ----
    if args.sweep:
        kappas = list(np.round(np.arange(0.0, 3.01, 0.1), 4))
        print(
            f"\nRunning kappa sweep: {len(kappas)} values "
            f"[{kappas[0]:.1f} … {kappas[-1]:.1f}], "
            f"season={args.season}, budget={args.budget_min:.0f} min"
        )
        rows = run_sweep(data, args.budget_min, kappas, args.method, args.season)

        # Print CSV
        header = "kappa,confidence_pct,total_rating,total_nominal,total_robust,n_rides,solver_status,rides_selected"
        print("\n" + header)
        for row in rows:
            print(
                f"{row['kappa']:.4f},{row['confidence_pct']:.2f},"
                f"{row['total_rating']:.2f},{row['total_nominal']:.2f},"
                f"{row['total_robust']:.2f},{row['n_rides']},"
                f"{row['solver_status']},{row['rides_selected']}"
            )

        if not args.no_save:
            output_dir.mkdir(parents=True, exist_ok=True)
            tag      = f"sweep__{args.season}__budget{args.budget_min:.0f}__{args.method}"
            csv_path = output_dir / f"{tag}.csv"

            with csv_path.open("w", encoding="utf-8") as f:
                f.write(header + "\n")
                for row in rows:
                    f.write(
                        f"{row['kappa']:.4f},{row['confidence_pct']:.2f},"
                        f"{row['total_rating']:.2f},{row['total_nominal']:.2f},"
                        f"{row['total_robust']:.2f},{row['n_rides']},"
                        f"{row['solver_status']},{row['rides_selected']}\n"
                    )
            print(f"\nSaved: {csv_path}")

            # Optional: plot if matplotlib available
            _try_plot_sweep(rows, args, output_dir, tag)

        return

    # ---- Single solve ----
    print(
        f"\nSolving: method={args.method}, season={args.season}, "
        f"budget={args.budget_min:.1f} min, "
        f"κ={kappa:.4f} (confidence={confidence_pct:.1f}%)"
    )

    x, status = _dispatch_solver(data, args.budget_min, kappa, args.method)

    if x is None:
        print(f"\n[ERROR] Solver returned no solution (status={status}).")
        sys.exit(1)

    # Build result object
    res = SOCPResult(
        method=args.method,
        season=args.season,
        budget=args.budget_min,
        kappa=kappa,
        confidence_pct=confidence_pct,
        solver_status=status,
        ride_names=data.ride_names,
        x=x,
        ratings=data.ratings,
        det_costs=data.det_costs,
        mu=data.mu,
    )
    res.set_robust_cost(data)

    # ---- Print ----
    print()
    print(f"{'='*70}")
    print(f"  ROBUST SOCP ITINERARY PLANNER")
    print(f"{'='*70}")
    print(f"  Season:       {args.season}")
    print(f"  Method:       {args.method}")
    print(f"  Budget:       {args.budget_min:.1f} min")
    print(f"  κ:            {kappa:.4f}  (confidence = {confidence_pct:.1f}%)")
    print(f"  Status:       {status}")
    print(f"{'='*70}")
    print()
    print(f"  Total rating:    {res.total_rating:.2f}")
    print(f"  Rides selected:  {res.n_rides}")
    print()
    print(f"  Time breakdown:")
    print(f"    Det. cost  (2·drive + duration):        {res.total_det_cost:.2f} min")
    print(f"    Nominal    (det + mean wait):           {res.total_nominal:.2f} min")
    print(f"    Robust     (nominal + κ·‖L^T x‖₂):     {res.total_robust:.2f} min")
    print(f"    Budget:                                 {res.budget:.2f} min")
    print(f"    Slack (budget − robust):                {res.budget - res.total_robust:.2f} min")
    print()
    print(f"  {'Key':<26}  {'Name':<34}  {'Rating':>6}  {'Det':>6}  {'μ wait':>7}  {'Sel':>4}")
    print(f"  {'-'*26}  {'-'*34}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*4}")

    for i, name in enumerate(data.ride_names):
        sel   = "✓" if x[i] > 0.5 else " "
        dname = ride_display.get(name, name)
        print(
            f"  {name:<26}  {dname:<34}  {data.ratings[i]:>6.1f}  "
            f"{data.det_costs[i]:>6.2f}  {data.mu[i]:>7.2f}  {sel:>4}"
        )

    print()
    print(f"  Rides picked (sorted by rating desc):")
    selected = sorted(
        [(data.ride_names[i], data.ratings[i], data.det_costs[i], data.mu[i])
         for i in range(data.n) if x[i] > 0.5],
        key=lambda t: -t[1],
    )
    for name, rating, det, mu_w in selected:
        dname = ride_display.get(name, name)
        print(f"    - {name:<26}  {dname:<34}  rating={rating:.1f}  cost={det + mu_w:.2f} (est. total={det + mu_w:.2f}) min")

    # ---- Save ----
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        kappa_tag = f"kappa{kappa:.3f}".replace(".", "p")
        conf_tag  = f"conf{confidence_pct:.0f}pct"
        tag       = f"socp__{args.method}__{args.season}__budget{args.budget_min:.0f}__{kappa_tag}__{conf_tag}"
        txt_path  = output_dir / f"{tag}.txt"

        lines = [
            f"Method:              {args.method}",
            f"Season:              {args.season}",
            f"Budget (min):        {args.budget_min:.2f}",
            f"Kappa:               {kappa:.4f}  (confidence = {confidence_pct:.1f}%)",
            f"Solver status:       {status}",
            "",
            f"Total rating:        {res.total_rating:.2f}",
            f"Rides selected:      {res.n_rides}",
            "",
            "Time breakdown:",
            f"  Det. cost  (2*drive + duration):        {res.total_det_cost:.2f} min",
            f"  Nominal    (det + mean wait):           {res.total_nominal:.2f} min",
            f"  Robust     (nominal + kappa*||L^T x||): {res.total_robust:.2f} min",
            f"  Budget:                                 {res.budget:.2f} min",
            f"  Slack (budget - robust):                {res.budget - res.total_robust:.2f} min",
            "",
            "Rides picked (in order of descending rating):",
        ]
        for name, rating, det, mu_w in selected:
            dname = ride_display.get(name, name)
            lines.append(
                f"  - {name:<26}  {dname:<34}  "
                f"rating={rating:.1f}  det={det:.2f}  mu_wait={mu_w:.2f}  "
                f"total_cost_est={det + mu_w:.2f} min"
            )

        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved: {txt_path}")


# ---------------------------------------------------------------------------
# Optional sweep plot
# ---------------------------------------------------------------------------

def _try_plot_sweep(rows: List[dict], args, output_dir: Path, tag: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    kappas    = [r["kappa"]          for r in rows]
    ratings   = [r["total_rating"]   for r in rows]
    nominals  = [r["total_nominal"]  for r in rows]
    robustes  = [r["total_robust"]   for r in rows]
    conf_pcts = [r["confidence_pct"] for r in rows]
    n_rides   = [r["n_rides"]        for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"Robust SOCP Sensitivity to κ\n"
        f"Season={args.season}, Budget={args.budget_min:.0f} min, Method={args.method}",
        fontsize=12,
    )

    ax1 = axes[0]
    ax1.plot(kappas, ratings, "o-", color="steelblue", label="Total rating")
    ax1.set_ylabel("Total Rating")
    ax1.set_ylim(0, max(ratings) * 1.15)
    ax1b = ax1.twinx()
    ax1b.plot(kappas, n_rides, "s--", color="coral", alpha=0.7, label="# rides")
    ax1b.set_ylabel("# Rides selected", color="coral")
    ax1b.tick_params(axis="y", labelcolor="coral")
    ax1.set_title("Objective vs κ")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(kappas, nominals, "o-",  color="green",  label="Nominal time (det + mean wait)")
    ax2.plot(kappas, robustes, "s--", color="orange", label="Robust time (+ κ·risk term)")
    ax2.axhline(args.budget_min, color="red", linestyle=":", linewidth=1.5, label=f"Budget ({args.budget_min:.0f} min)")
    ax2.set_ylabel("Time (min)")
    ax2.set_xlabel("κ (robustness)")
    ax2.legend(fontsize=8)
    ax2.set_title("Time Budget Utilisation vs κ")
    ax2.grid(True, alpha=0.3)

    # Add secondary x-axis with confidence percentages
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    tick_kappas = [0.0, 0.842, 1.282, 1.645, 2.0, 2.326, 3.0]
    tick_labels = ["50%", "80%", "90%", "95%", "97.7%", "99%", "99.9%"]
    valid_ticks = [(k, l) for k, l in zip(tick_kappas, tick_labels) if k <= max(kappas)]
    if valid_ticks:
        ax2_top.set_xticks([t[0] for t in valid_ticks])
        ax2_top.set_xticklabels([t[1] for t in valid_ticks], fontsize=8)
        ax2_top.set_xlabel("Confidence level", fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / f"{tag}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
