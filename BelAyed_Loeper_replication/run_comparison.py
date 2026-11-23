#!/usr/bin/env python3
"""Paired FHC experiment: unchanged Loeper reproduction versus Q=I linear solve.

Start from main.py or import run_fhc_comparison.
All plots, tables and metadata are in risultati. No stochastic volatility
regime or new market dynamics are introduced. The synthetic Dupire surface
is the existing reproduction's default, not the authors' proprietary data.
"""

from __future__ import annotations
import csv
import gzip
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vega_hedging_reproduction as core
from adapted_simulator import simulate_fhc

CASES = {
    "constant": ([0], "Constant"),
    "exponential_05": ([1], "Exponential (lambda=0.5)"),
    "constant_exponential_05": ([0, 1], "Constant + exponential (0.5)"),
    "self_similar_spot": ([2], "Self-similar (lambda=1, nu=0.01)"),
    "constant_self_similar_spot": ([0, 2], "Constant + self-similar (1, 0.01)"),
}


def log(text):
    """Print a progress message immediately, including during long Monte Carlo runs."""
    print(text, flush=True)


def save_grid(path, grid):
    """Save both the regular and post-event PDE values in a compressed NumPy file."""
    fields = {
        "time": grid.t,
        "spot": grid.s,
        "values": grid.values,
        "event_indices": np.array(sorted(grid.post_events), dtype=int),
    }
    for k, v in grid.post_events.items():
        fields["post_" + str(k)] = v
    np.savez_compressed(path, **fields)


def load_grid(path):
    """Restore a Grid with its time axis, spot axis and post-event values."""
    with np.load(path) as f:
        return core.Grid(
            f["time"],
            f["spot"],
            f["values"],
            {int(k): f["post_" + str(k)] for k in f["event_indices"]},
        )


class AuditTrace:
    """All-path daily cost totals; individual decisions for paths 0..15 only."""

    def __init__(self, out, name, n, risk, indices):
        """Prepare daily cost totals and trace the first sixteen paths in detail."""
        self.out = out
        self.name = name
        self.n = n
        self.risk = risk
        self.indices = indices
        self.last = np.zeros((n, 2))
        self.daily = np.zeros((1261, 2))
        self.max_abs_weight = 0.0
        self.events = 0
        self.initial = None
        self.trace = core.CsvMcTrace(
            out / (name + "_trace_16paths.csv.gz"), [f"put_{k}" for k in range(30, 161, 10)]
        )

    def __enter__(self):
        """Open the trades and sensitivity traces and write their column headers."""
        self.trace.__enter__()
        self.file = gzip.open(
            self.out / (self.name + "_sensitivities_16paths.csv.gz"), "wt", newline=""
        )
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "path_id",
                "day",
                "spot",
                "risk_index",
                "target_b",
                "residual_before",
                "residual_after",
            ]
            + [f"M_put_{k}" for k in range(30, 161, 10)]
        )
        return self

    def __exit__(self, *args):
        """Close both trace files at the end of a simulation."""
        self.file.close()
        self.trace.__exit__(*args)

    def emit(self, ids, day, spots, event, alive, tracked, q, q2, cq, cq2, count, w, dw):
        """Accumulate daily costs and record selected paths' trades and hedge residuals."""
        ids = np.asarray(ids)
        n = len(ids)
        if not n:
            return
        new = np.column_stack([np.broadcast_to(cq, (n,)), np.broadcast_to(cq2, (n,))])
        self.daily[day] += np.sum(new - self.last[ids], axis=0)
        self.last[ids] = new
        self.max_abs_weight = max(self.max_abs_weight, float(np.max(np.abs(w))))
        self.events += n
        if event == 0:
            self.initial = np.asarray(w[0]).copy()
        select = ids < 16
        if not np.any(select):
            return
        v = lambda x: np.broadcast_to(np.asarray(x), (n,))[select]
        self.trace.emit(
            ids[select],
            day,
            v(spots),
            event,
            v(alive),
            v(tracked),
            v(q),
            v(q2),
            v(cq),
            v(cq2),
            v(count),
            w[select],
            dw[select],
        )
        if event in (0, 1):
            sampled = self.risk.sample(day / 252, v(spots))
            for j, pid in enumerate(ids[select]):
                for ix in self.indices:
                    A = sampled[j, ix, 1:]
                    b = sampled[j, ix, 0]
                    after = w[select][j]
                    before = after - dw[select][j]
                    self.writer.writerow(
                        [pid, day, v(spots)[j], ix, b, b - A @ before, b - A @ after, *A]
                    )


def export_snapshots(out, base, risk, model):
    # Complete daily local-vol surfaces, with log-moneyness and explicit S=0.
    """Save the local-volatility surfaces and selected PDE price and sensitivity slices."""
    with gzip.open(out / "local_volatility_surfaces.csv.gz", "wt") as f:
        f.write("time,spot,log_moneyness,base,parallel,exponential_05,self_similar_spot\n")
        positive = base.s > 0
        for t in base.t:
            s = base.s[positive]
            lv = model["base"](t, s)
            np.savetxt(
                f,
                np.column_stack(
                    [
                        np.full(len(s), t),
                        s,
                        np.log(s / 100),
                        lv,
                        lv + 0.01,
                        lv + 0.01 * np.exp(-0.5 * t),
                        lv + 0.01 * np.exp(-t - 0.01 * np.log(s / 100)),
                    ]
                ),
                fmt="%.17g",
                delimiter=",",
            )
    # Full spot rows at declared snapshot dates. Full time grids are in NPZ.
    times = [0, 0.5, 1 - 1 / 252, 1, 1 + 1 / 252, 2, 3, 4, 5 - 1 / 252, 5]
    with gzip.open(out / "pde_sensitivity_snapshots.csv.gz", "wt") as f:
        f.write(
            "time,spot,product_id,base_price,parallel_price_change,exponential_05_price_change,self_similar_spot_price_change\n"
        )
        for t in times:
            prices = base.sample(t, base.s)
            sens = risk.sample(t, base.s)
            blocks = []
            for j in range(15):
                blocks.append(
                    np.column_stack(
                        [
                            np.full(len(base.s), t),
                            base.s,
                            np.full(len(base.s), j),
                            prices[:, j],
                            sens[:, :, j],
                        ]
                    )
                )
            np.savetxt(f, np.concatenate(blocks), fmt="%.17g", delimiter=",")


def plot_results(out, rows, curves, pathwise):
    """Plot FHC estimates, cumulative costs and empirical pathwise cost distributions."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = [CASES[k][1] for k in CASES]
    x = np.arange(len(labels))
    colors = ["#245a81", "#e18c35"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    for ax, cost, title in zip(
        axs, ["constant_spread", "linear_impact"], ["Constant spread", "Linear price impact"]
    ):
        for i, solver in enumerate(["loeper", "quadratic"]):
            rr = [r for r in rows if r["solver"] == solver and r["cost_model"] == cost]
            ax.errorbar(
                [r["mean_bp"] for r in rr],
                x + (i - 0.5) * 0.24,
                xerr=[1.96 * r["se_bp"] for r in rr],
                color=colors[i],
                fmt="o" if i == 0 else "x",
                capsize=3,
                markersize=6,
                label="Loeper / SVD" if i == 0 else "Linear solve / Q=I",
                alpha=0.9,
            )
        ax.set_yticks(x, labels if ax is axs[0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_xlabel("Future hedging cost (bp, logarithmic axis)")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Same market paths and constraints: paired algorithm comparison\nExisting synthetic local volatility; error bars show Monte Carlo 95% intervals",
        fontsize=12,
    )
    fig.savefig(out / "fhc_comparison.png", dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for key in ["constant", "exponential_05", "constant_exponential_05"]:
        for solver, style in [("loeper", "-"), ("quadratic", "--")]:
            for j in range(2):
                axs[j].plot(
                    np.arange(1261) / 252,
                    curves[(key, solver)][:, j],
                    style,
                    label=CASES[key][1] + (" / SVD" if solver == "loeper" else " / linear"),
                    lw=1.6,
                )
    for ax, title in zip(axs, ["Constant spread", "Linear price impact"]):
        ax.set(title=title, xlabel="Elapsed time (years)", ylabel="Mean accumulated cost (bp)")
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=7)
    fig.suptitle("Accumulated hedging costs, including early-call unwind")
    fig.savefig(out / "fhc_accumulation.png", dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, key in zip(axs, ["exponential_05", "constant_exponential_05"]):
        for solver, style in [("loeper", "-"), ("quadratic", "--")]:
            values = pathwise[(key, solver)][:, 0]
            xs = np.sort(values)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), style, label=solver, lw=1.7)
        ax.set(
            title=CASES[key][1],
            xlabel="Pathwise constant-spread cost (bp)",
            ylabel="Empirical CDF",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Self-similar deformation: full cost distributions")
    fig.savefig(out / "fhc_distribution.png", dpi=180)
    plt.close(fig)


def load_or_solve_grids(out, products, model, cfg, source_hash):
    """Load compatible price and sensitivity grids, or solve and save them once."""
    # Both distributed editions use identical numerical PDE routines. Preserve
    # the earlier cache key only for this verified source hash.
    if source_hash == "06b130e09b29c07b11fa2949e305e7b190a50f43ba21a7e1cc2892046611badc":
        source_hash = "6776663e51164394a16587a8bbe26233e3fc1c2ac4daa446edf0dfb1ba704050"
    fingerprint = hashlib.sha256(
        json.dumps(
            {"ds": cfg.ds, "steps": cfg.steps_year, "smax": cfg.smax, "source": source_hash},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = out / "pde_cache"
    cache.mkdir(exist_ok=True)
    cache_complete = all(
        (cache / name).is_file()
        for name in ["fingerprint.txt", "base_prices.npz", "price_changes.npz"]
    )
    if cache_complete and (cache / "fingerprint.txt").read_text() == fingerprint:
        log("Loading matching PDE cache.")
        base = load_grid(cache / "base_prices.npz")
        risk = load_grid(cache / "price_changes.npz")
    else:
        log(
            "Solving unchanged PDE engine: central, parallel, exp(-0.5t), exp(-t-0.01 log(S/100))."
        )
        sigmas = [model["base"]] + [
            core.deformation(model["base"], 0.01, lam, nu)
            for lam, nu in [(0, 0), (0.5, 0), (1, 0.01)]
        ]
        base, risk = core.risk_bundle(products, sigmas, cfg)
        log("Saving full PDE price and sensitivity grids.")
        save_grid(cache / "base_prices.npz", base)
        save_grid(cache / "price_changes.npz", risk)
        (cache / "fingerprint.txt").write_text(fingerprint)
    return base, risk


def check_simulator(out, risk, cfg, model):
    """Check the adapted simulator against the reference on 256 common paths."""
    log("Checking adapted simulator against the unchanged source, same paths.")
    original = core.simulate_fhc(
        risk, cfg, npaths=256, seed=cfg.seed + 11, factors=2, sigma=model["base"]
    )
    adapted = simulate_fhc(
        risk, cfg, npaths=256, seed=cfg.seed + 11, risk_indices=[0, 1], sigma=model["base"]
    )
    check = {
        key: float(np.max(np.abs(original[key] - adapted[key])))
        for key in ["q", "q2", "duration", "rehedges"]
    }
    if max(check.values()) != 0:
        raise ArithmeticError(f"Simulator differs from replication: {check}")
    core.write_json(out / "simulator_equivalence.json", check)


def estimate_qstar(risk, cfg, model, pilot_paths):
    """Estimate the typical daily change in exotic vega on an independent pilot."""
    log("Independent pilot for the original aggregate linear-impact cost model.")
    pilot = core.simulate_fhc(
        risk,
        cfg,
        npaths=pilot_paths,
        seed=cfg.seed + 901,
        factors=1,
        calibrate=True,
        sigma=model["base"],
    )
    qstar = pilot["qstar"]
    if qstar is None or qstar <= 0:
        raise ValueError("The pilot did not produce a positive Qstar.")
    return qstar


def path_costs_bp(simulation, spread, qstar):
    """Convert each path's traded vega to constant-spread and linear-impact FHC in bp."""
    constant_spread = spread * simulation["q"] * 1e4
    linear_impact = spread * simulation["q2"] / qstar * 1e4
    return np.column_stack([constant_spread, linear_impact])


def summarize_costs(case, solver, costs):
    """Summarize each cost convention across paths for one deformation and hedge rule."""
    rows = []
    for j, cost in enumerate(["constant_spread", "linear_impact"]):
        summary = core.stats(costs[:, j])
        rows.append(
            dict(
                case=case,
                solver=solver,
                cost_model=cost,
                mean_bp=summary["mean"],
                se_bp=summary["standard_error"],
                ci95_low_bp=summary["ci95_low"],
                ci95_high_bp=summary["ci95_high"],
                median_bp=summary["median"],
                p95_bp=summary["p95"],
                paths=len(costs),
            )
        )
    return rows


def paired_cost_differences(case, reference_costs, quadratic_costs):
    """Summarize the path-matched differences in spread and impact costs for one case."""
    paired = []
    diff = quadratic_costs - reference_costs
    for j, cost in enumerate(["constant_spread", "linear_impact"]):
        summary = core.stats(diff[:, j])
        paired.append(
            dict(
                case=case,
                cost_model=cost,
                mean_difference_bp=summary["mean"],
                se_difference_bp=summary["standard_error"],
                max_abs_path_difference_bp=float(np.max(np.abs(diff[:, j]))),
                ci95_low_bp=summary["ci95_low"],
                ci95_high_bp=summary["ci95_high"],
            )
        )
    return paired


def experiment_metadata(cfg, products, model):
    """Record the contracts, volatility bumps, numerical settings and cost conventions."""
    return {
        "config": asdict(cfg),
        "Q": "identity; no alternative economic matrix assumed",
        "deformations": [
            {"epsilon": 0.01, "lambda": 0.0, "nu": 0.0},
            {"epsilon": 0.01, "lambda": 0.5, "nu": 0.0},
            {"epsilon": 0.01, "lambda": 1.0, "nu": 0.01},
        ],
        "anchor": {"time": 0.0, "spot": 100.0},
        "model": model["metadata"],
        "products": [asdict(p) for p in products],
        "scope": "Algorithm comparison under the unchanged reproduction synthetic LV. Not the paper market calibration.",
        "source_sha256": core.sha256_file(Path(core.__file__)),
        "forward_bumps": "Raw price differences, anchored at t=0,S=100, as in source Algorithm 1 implementation.",
        "cost_conventions": "spread=0.002 on parallel decimal vega; initial cost excluded; early-call unwind included; no natural-expiry cost.",
        "trace_scope": "Paths 0..15, all actual decisions; all paths have final cost exports.",
        "pde_csv_scope": "Full spot axis at listed snapshots; complete daily price/risk grids in NPZ.",
        "self_similarity_note": "Anchor changes rescale exact directional derivatives, not finite bumps exactly; no time re-anchoring is introduced.",
    }


def run_fhc_comparison(
    output_dir,
    *,
    paths=10_000,
    pilot_paths=None,
    ds=0.5,
    steps_year=252,
    seed=4550643,
    cases=None,
    export_grids=True,
):
    """Compare the two hedge rules on identical paths and write paired costs in basis points.

    paths controls the common test paths. pilot_paths sets the independent cost
    calibration sample and defaults to paths. cases defaults to all five scenarios.
    """
    if pilot_paths is None:
        pilot_paths = paths
    cases = list(CASES) if cases is None else list(cases)
    if not cases or len(cases) != len(set(cases)) or any(key not in CASES for key in cases):
        raise ValueError(f"Choose one or more distinct cases from {list(CASES)}.")
    if min(paths, pilot_paths) < 2:
        raise ValueError("Use at least two paths for both the comparison and the pilot.")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = core.Config(
        ds=ds, steps_year=steps_year, paths=paths, seed=seed, csv_compression="gzip"
    )
    model = core.build_dupire_market(out)
    products = [core.Product("autocall", 5, coupon=0.03, coupon_mode="flat")] + [
        core.Product("put", 5, k) for k in range(30, 161, 10)
    ]
    metadata = experiment_metadata(cfg, products, model)
    base, risk = load_or_solve_grids(out, products, model, cfg, metadata["source_sha256"])
    metadata["initial_note_price"] = float(base.sample(0, [100])[0, 0])
    core.write_json(out / "experiment.json", metadata)
    check_simulator(out, risk, cfg, model)
    qstar = estimate_qstar(risk, cfg, model, pilot_paths)
    metadata.update(qstar=qstar, pilot_paths=pilot_paths, pilot_seed=cfg.seed + 901)
    log(f"Pilot Qstar={qstar:.10g}; running paired policies.")
    rows = []
    paired = []
    curves = {}
    pathwise = {}
    diagnostics = {}
    initial = []
    for key in cases:
        indices, _ = CASES[key]
        initial_sensitivities = risk.sample(0, [100])[0]
        M0 = initial_sensitivities[indices, 1:]
        singular = np.linalg.svd(M0, compute_uv=False)
        for solver in ["loeper", "quadratic"]:
            started = time.monotonic()
            log(f"{key}: {solver}, {paths} paths")
            with AuditTrace(out, key + "_" + solver, paths, risk, indices) as trace:
                sim = simulate_fhc(
                    risk,
                    cfg,
                    risk_indices=indices,
                    hedge_solver=solver,
                    sigma=model["base"],
                    trace=trace,
                )
            costs = path_costs_bp(sim, cfg.spread, qstar)
            pathwise[key, solver] = costs
            curves[key, solver] = (
                np.cumsum(trace.daily, axis=0)
                / paths
                * np.array([cfg.spread * 1e4, cfg.spread / qstar * 1e4])
            )
            assert np.allclose(
                curves[key, solver][-1], costs.mean(axis=0), atol=1e-8, rtol=1e-10
            )
            rows.extend(summarize_costs(key, solver, costs))
            np.savetxt(
                out / (key + "_" + solver + "_pathwise.csv.gz"),
                np.column_stack([np.arange(paths), costs, sim["duration"], sim["rehedges"]]),
                delimiter=",",
                header="path_id,constant_spread_bp,linear_impact_bp,duration_years,rehedges",
                comments="",
            )
            diagnostics[key + "_" + solver] = {
                "max_post_update_residual": sim["post_residual_max"],
                "threshold_breaches_after_update": sim["threshold_breaches_after_update"],
                "outside_grid_count": sim["outside_count"],
                "max_abs_weight": trace.max_abs_weight,
                "initial_singular_values": singular.tolist(),
                "elapsed_seconds": time.monotonic() - started,
            }
            for j, w in enumerate(trace.initial):
                initial.append([key, solver, 30 + 10 * j, w])
            log(
                f"  FHC spread={costs[:,0].mean():.6f} bp; impact={costs[:,1].mean():.6f} bp; max residual={sim['post_residual_max']:.3g}"
            )
        paired.extend(
            paired_cost_differences(key, pathwise[key, "loeper"], pathwise[key, "quadratic"])
        )
        core.write_csv(out / "fhc_summary.csv", list(rows[0]), [list(r.values()) for r in rows])
        core.write_csv(
            out / "paired_differences.csv", list(paired[0]), [list(r.values()) for r in paired]
        )
        core.write_json(out / "solver_diagnostics.json", diagnostics)
    core.write_csv(
        out / "initial_weights.csv", ["case", "solver", "put_strike", "weight"], initial
    )
    core.write_json(out / "experiment.json", metadata)
    curve_rows = (
        [key, solver, day, day / 252, *curve[day]]
        for (key, solver), curve in curves.items()
        for day in range(1261)
    )
    core.write_csv(
        out / "accumulated_costs.csv",
        ["case", "solver", "day", "time", "constant_spread_bp", "linear_impact_bp"],
        curve_rows,
    )
    if export_grids:
        log("Exporting volatility surfaces and declared PDE/sensitivity snapshots.")
        export_snapshots(out, base, risk, model)
    if cases == list(CASES):
        plot_results(out, rows, curves, pathwise)
    log("Completed. Results: " + str(out.resolve()))
