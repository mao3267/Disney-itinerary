from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch


def load_data(repo_root: Path):
    summary_csv = repo_root / "outputs" / "baselines" / "summary.csv"
    sweep_csv = repo_root / "outputs" / "socp" / "sweep__all__budget600__enumerate.csv"

    summary = pd.read_csv(summary_csv)
    sweep = pd.read_csv(sweep_csv)

    summary["baseline"] = summary["baseline"].astype(str).str.strip()
    summary["wait_type"] = summary["wait_type"].astype(str).str.strip()
    sweep.columns = [c.strip() for c in sweep.columns]

    return summary, sweep


def pick_existing_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot find any of these columns: {candidates}")


def get_kappa_zero_row(sweep: pd.DataFrame, tol: float = 1e-9):

    exact = sweep[(sweep["kappa"] - 0.0).abs() <= tol].copy()
    if not exact.empty:
        if "total_rating" in exact.columns:
            return exact.loc[exact["total_rating"].idxmax()]
        return exact.iloc[0]

    idx = (sweep["kappa"] - 0.0).abs().idxmin()
    row = sweep.loc[idx]
    print(f"[warn] exact kappa=0 not found; using nearest kappa={row['kappa']}")
    return row

def split_ride_list(s: str):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def build_selection_matrix(summary: pd.DataFrame, sweep: pd.DataFrame, wait_type: str = "all"):
    baseline_order = ["random", "greedy_rating", "max_count", "max_rating"]

    df = summary[summary["wait_type"] == wait_type].copy()
    if df.empty:
        raise ValueError(f"No baseline rows found for wait_type={wait_type}")

    df["baseline"] = pd.Categorical(df["baseline"], categories=baseline_order, ordered=True)
    df = df.sort_values("baseline")

    k0 = get_kappa_zero_row(sweep)

    method_to_rides = {}

    for _, row in df.iterrows():
        method_to_rides[str(row["baseline"])] = split_ride_list(row["rides_list"])

    robust_col = pick_existing_column(sweep, ["rides_selected", "rides_list"])
    method_to_rides["robust_kappa0"] = split_ride_list(k0[robust_col])

    all_rides = sorted(
        {
            ride
            for rides in method_to_rides.values()
            for ride in rides
        }
    )

    matrix = pd.DataFrame(0, index=all_rides, columns=list(method_to_rides.keys()), dtype=int)

    for method, rides in method_to_rides.items():
        for ride in rides:
            if ride in matrix.index:
                matrix.loc[ride, method] = 1

    # sort rides by how often they are selected
    matrix["selection_count"] = matrix.sum(axis=1)
    matrix = matrix.sort_values(["selection_count"], ascending=False).drop(columns=["selection_count"])

    return matrix


def make_figure2(summary: pd.DataFrame, sweep: pd.DataFrame, outdir: Path, wait_type: str = "all"):
    df = summary[summary["wait_type"] == wait_type].copy()

    baseline_order = ["random", "greedy_rating", "max_count", "max_rating"]
    df["baseline"] = pd.Categorical(df["baseline"], categories=baseline_order, ordered=True)
    df = df.sort_values("baseline")

    k0 = get_kappa_zero_row(sweep)

    colors = {
        "random": "#7f7f7f",
        "greedy_rating": "#1f77b4",
        "max_count": "#ff7f0e",
        "max_rating": "#2ca02c",
        "robust_kappa0": "#d62728",
    }


    legend_elements = [
        Patch(facecolor=c, label=m) for m, c in colors.items()
    ]

    rating_col = pick_existing_column(sweep, ["total_rating", "objective", "obj"])
    rides_col = pick_existing_column(sweep, ["n_rides", "rides_count", "ride_count"])
    time_col = pick_existing_column(sweep, ["total_nominal", "nominal_time", "total_time_used_min"])

    plot_df = pd.DataFrame({
        "method": list(df["baseline"].astype(str)) + ["robust_kappa0"],
        "total_rating": list(df["total_rating"]) + [k0[rating_col]],
        "rides_count": list(df["rides_count"]) + [k0[rides_col]],
        "time_used": list(df["total_time_used_min"]) + [k0[time_col]],
    })

    bar_colors = [colors[m] for m in plot_df["method"]]

    x = list(range(len(plot_df)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"Baseline comparison\nSeason={wait_type}, Budget=600 min, robust reference=$\\kappa=0$",
        fontsize=18,
        y=0.98,
    )

    axes[0].legend(handles=legend_elements, fontsize=11)

    axes[0].bar(x, plot_df["total_rating"], color = bar_colors)
    axes[0].set_ylabel("Total Rating", fontsize=14)
    axes[0].set_title("Objective comparison", fontsize=16, pad=10)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, plot_df["rides_count"], color = bar_colors)
    axes[1].set_ylabel("# Rides", fontsize=14)
    axes[1].set_title("Ride count comparison", fontsize=16, pad=10)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, plot_df["time_used"], color = bar_colors)
    axes[2].axhline(600, linestyle=":", linewidth=2, color="red", label="Budget (600 min)")
    axes[2].set_ylabel("Time Used (min)", fontsize=14)
    axes[2].set_title("Time budget utilization", fontsize=16, pad=10)
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=12)

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(plot_df["method"], rotation=15, fontsize=12)
    axes[2].set_xlabel("Method", fontsize=14)

    outpath = outdir / f"baseline_comparison__{wait_type}.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")


def make_figure3(summary: pd.DataFrame, outdir: Path):
    df = summary.copy()

    season_order = ["value", "regular", "peak", "all"]
    baseline_order = ["random", "greedy_rating", "max_count", "max_rating"]

    df["wait_type"] = pd.Categorical(df["wait_type"], categories=season_order, ordered=True)
    df["baseline"] = pd.Categorical(df["baseline"], categories=baseline_order, ordered=True)
    df = df.sort_values(["baseline", "wait_type"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle("Season comparison", fontsize=18, y=0.98)

    for baseline in baseline_order:
        sub = df[df["baseline"] == baseline].copy()
        if sub.empty:
            continue

        axes[0].plot(sub["wait_type"], sub["total_rating"], marker="o", linewidth=2, label=baseline)
        axes[1].plot(sub["wait_type"], sub["rides_count"], marker="o", linewidth=2, label=baseline)
        axes[2].plot(sub["wait_type"], sub["total_time_used_min"], marker="o", linewidth=2, label=baseline)

    axes[0].set_ylabel("Total Rating", fontsize=14)
    axes[0].set_title("Objective vs season", fontsize=16, pad=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=11, ncol=2)

    axes[1].set_ylabel("# Rides", fontsize=14)
    axes[1].set_title("Ride count vs season", fontsize=16, pad=10)
    axes[1].grid(alpha=0.3)

    axes[2].set_ylabel("Time Used (min)", fontsize=14)
    axes[2].set_title("Time budget utilization vs season", fontsize=16, pad=10)
    axes[2].axhline(600, linestyle=":", linewidth=2, color="red", label="Budget (600 min)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=11)

    axes[2].set_xlabel("Season / wait assumption", fontsize=14)

    outpath = outdir / "season_comparison.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")


def make_figure4(
    summary: pd.DataFrame,
    sweep: pd.DataFrame,
    outdir: Path,
    wait_type: str = "all",
):
    matrix = build_selection_matrix(summary, sweep, wait_type=wait_type)

    fig, ax = plt.subplots(figsize=(8, max(6, 0.45 * len(matrix.index))))

    im = ax.imshow(matrix.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=20, fontsize=11)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=10)

    ax.set_title(
        f"Ride selection heatmap\nSeason={wait_type}, robust reference=$\\kappa=0$",
        fontsize=16,
        pad=12,
    )

    # annotate each cell with 0/1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix.iloc[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="black" if val == 0 else "white",
                fontsize=9,
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Selected (1) / Not selected (0)", fontsize=11)

    plt.tight_layout()

    outpath = outdir / f"ride_selection_heatmap__{wait_type}.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")

def make_figure5(
    summary: pd.DataFrame,
    sweep: pd.DataFrame,
    outdir: Path,
    wait_type: str = "all",
):
    matrix = build_selection_matrix(summary, sweep, wait_type=wait_type)
    freq = matrix.sum(axis=1).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.4 * len(freq.index))))
    ax.barh(freq.index, freq.values)
    ax.invert_yaxis()

    ax.set_xlabel("Number of methods selecting the ride", fontsize=12)
    ax.set_ylabel("Ride", fontsize=12)
    ax.set_title(
        f"Ride selection frequency\nSeason={wait_type}, including robust reference $\\kappa=0$",
        fontsize=16,
        pad=12,
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    outpath = outdir / f"ride_selection_frequency__{wait_type}.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {outpath}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    outdir = repo_root / "outputs" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    summary, sweep = load_data(repo_root)

    # make_figure2(summary, sweep, outdir, wait_type="all")
    # make_figure3(summary, outdir)

    # make_figure4(summary, sweep, outdir, wait_type="all")
    # make_figure5(summary, sweep, outdir, wait_type="all")


if __name__ == "__main__":
    main()