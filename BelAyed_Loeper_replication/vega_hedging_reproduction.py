#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifiable reproduction of Bel Hadj Ayed & Loeper (24 August 2023).

Paper: The Cost of Vega-Hedging Structured Products, SSRN 4550643.
Python >= 3.10; numpy, scipy, matplotlib; PyMuPDF for the original figures.

Start from main.py. For a custom run, import run_replication and Config.
Section 3 prices products on PDE grids and computes finite-bump sensitivities.
Section 4 estimates future hedging costs on simulated local-volatility paths.
The paper's proprietary surface is unavailable. The default Section 4 run
uses synthetic SSVI quotes and Dupire local volatility. A supplied CSV can
replace those quotes. Figure 6 is extracted only if the source PDF is supplied.
See ASSUMPTIONS for the numerical conventions and reproduction limits.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import gc
import gzip
import hashlib
import json
import platform
import time
import warnings

try:
    import numpy as np
    import scipy
    from scipy.linalg import solve_banded
    from scipy.optimize import brentq
    from scipy.special import ndtr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Run: python -m pip install numpy scipy matplotlib PyMuPDF"
    ) from exc

SOURCE_URL = "https://ssrn.com/abstract=4550643"
ASSUMPTIONS = [
    "r=q=0: interest rates and dividends are not specified in the paper; costs are not discounted.",
    "Initial spot 100; vanilla prices in index points; note notional 1.",
    "PDE: uniform S grid, Crank-Nicolson, two implicit Euler half-steps after payoffs/events (Rannacher smoothing).",
    "Boundaries: asymptotic values at S=0 and S=Smax; no separately calibrated boundary model.",
    "Cash digital pays 1; terminal value 0.5 exactly at the strike (cell averaging).",
    "Autocall PDE: average the left and right limits at jump nodes ONLY for backward propagation; event grids and Monte Carlo retain the exact contractual redemption.",
    "Autocall: annual observations with S>=100, FINAL European barrier S<70; payoff S/100 below 70, 1 between 70 and 100, and 1+coupon above 100.",
    "The monitoring convention and barrier formula in Figure 6 are ambiguous; the final European barrier follows the text of Section 3.4.",
    "Section 3: snowball coupons 10/20/30%; Section 4: FIXED 3% rebate by default; coupon_mode='snowball' applies 3% per elapsed year.",
    "Section 3 bumps: epsilon=0.01*sigma (0.003 at 30%, 0.002 at 20%); Section 4: ABSOLUTE parallel bump 0.01.",
    "Sensitivities use FORWARD bumps as in Algorithm 1; central differences do not replace the paper's method.",
    "Two purely time-dependent shifts and same-maturity vanilla options have rank 1 in the infinitesimal limit: the second rank under finite bumps depends on nonlinearity.",
    "SVD pseudoinverse with configurable rcond; no cap on weights; unidentifiable points are masked and diagnostics are saved.",
    "Theoretical figures: absolute minimum singular value 1e-8 for identifiability; robustness ratios displayed only for |price change|>=1e-6 (Figure 5) or 1e-5 (Figure 9). Source grids and ratios before the denominator mask are saved.",
    "Source discrepancy: Figure 5(c) in the PDF shows mostly positive residuals; with the printed formula and parameters, the code and an independent quadrature give negative values. The sign has not been changed.",
    "Robustness bumps are anchored at t0=0, S0=100, consistent with Algorithm 1 grids; log(S/100) is regularized only at the S=0 boundary.",
    "The 0.1 bp threshold applies to the maximum residual of PRICE CHANGES under the bumps, not to derivatives per unit of volatility.",
    "MC: 252 business days/year; daily log-Euler for LV; exact lognormal steps for constant sigma; explicit seed, no antithetic sampling.",
    "Prices and finite differences are interpolated linearly BEFORE solving the hedge; states before and after an autocall event are stored separately.",
    "Initial hedge w=A^+b; subsequent hedges w_new=w_old+A^+(b-Aw_old); initial hedging cost is excluded.",
    "Hedge liquidation at early autocall is included by default; include_unwind=False excludes it. No transaction cost is charged at the natural expiry of vanilla options.",
    "The 20 bp spread is interpreted as the mid-to-execution distance in absolute volatility: cost=0.002*gross_decimal_vega.",
    "Cost vega: response to the selected parallel scenario divided by 0.01 (local by default; recalibrated implied with deformation_space='implied'); this explicit convention is not fully specified by the authors.",
    "Aggregate quadratic cost: 0.002*Q^2/Qstar, Q=sum_j |dw_j|*|vega_j|; not the sum of squared individual-instrument amounts.",
    "Qstar is the mean absolute daily change in exotic vega over paths alive at the start of the step, including termination; estimated with an independent pilot run.",
    "Qstar remains fixed when hedging frequency changes; option contracts and autocall observation dates are unchanged by the rehedging frequency.",
    "FHC(t,S): a fresh valuation conditional on the note being alive; the initial cost of the new hedge is excluded also at t>0.",
    "MC beyond Smax: asymptotic sensitivities are zero and occurrences are counted in the report; simulated paths are not clipped.",
    "Monte Carlo intervals: mean +/-1.96*SE, descriptive; they exclude PDE, model and calibration errors.",
    "FHC curves are evaluated at a declared number of points, connected for display; original chart values are not digitized.",
    "Dupire extension: deterministic local volatility extracted from total implied variance, not a two-factor stochastic local-volatility model.",
    "Dupire CSV: maturity in years, strike in the units of the supplied spot, implied_vol in decimals; a complete rectangular grid. The engine retains r=q=0.",
    "Dupire: cubic interpolation of total variance; analytic spline derivatives; numerical calendar and butterfly checks on a dense grid, without a global guarantee.",
    "Dupire: explicit continuation before the first maturity and outside the strike range; negative variances are not automatically repaired. Vanilla repricing is saved separately.",
    "Default deformations act on LOCAL volatility to retain the paper's choice; deformation_space='implied' permits implied-volatility bumps followed by a new Dupire extraction.",
]


def log(message):
    """Print a progress message immediately, including during long Monte Carlo runs."""
    print(message, flush=True)


def write_json(path, obj):
    """Write a JSON report, converting NumPy values and paths to portable types."""

    def convert(x):
        """Convert an array, scalar or file path to a JSON-compatible value."""
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    Path(path).write_text(
        json.dumps(obj, default=convert, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path, columns, rows):
    """Write column names and data rows to a UTF-8 CSV file."""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def sha256_file(path):
    """Hash large files in blocks without loading the entire CSV into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Config:
    """PDE resolution, Monte Carlo settings, hedge tolerances and export options."""

    ds: float = 0.5
    steps_year: int = 252
    smax: float = 600.0
    paths: int = 10000
    profile_paths: int = 10000
    seed: int = 4550643
    rcond: float = 1e-10
    min_scenario_price: float = 1e-8
    loss_tolerance: float = 1e-5
    spread: float = 0.002
    coupon_mode: str = "flat"
    include_unwind: bool = True
    dpi: int = 150
    profile_spots: int = 17
    profile_times_year: int = 4
    csv_compression: str = "none"
    trace_runs: str = "main"


@dataclass(frozen=True)
class Product:
    """Contract terms used to set terminal payoffs, coupons and PDE boundaries."""

    kind: str
    maturity: int
    strike: float = 100.0
    strike2: float = 110.0
    barrier: float = 70.0
    coupon: float = 0.10
    coupon_mode: str = "snowball"

    def redemption(self, year):
        """Return the autocall redemption for a given observation year."""
        return 1.0 + self.coupon * (year if self.coupon_mode == "snowball" else 1.0)

    def payoff(self, s):
        """Evaluate the terminal payoff at each supplied spot."""
        if self.kind in ("call", "upout"):
            return np.maximum(s - self.strike, 0.0)
        if self.kind == "put":
            return np.maximum(self.strike - s, 0.0)
        if self.kind == "spread":
            return np.maximum(s - self.strike, 0) - np.maximum(s - self.strike2, 0)
        if self.kind == "digital":
            return (s > self.strike).astype(float) + 0.5 * (s == self.strike)
        if self.kind == "autocall":
            return np.where(
                s >= 100,
                self.redemption(self.maturity),
                np.where(s < self.barrier, s / 100, 1.0),
            )
        raise ValueError(self.kind)

    def boundaries(self, t, smax, side="pre"):
        """Return lower and upper boundary prices before or after a contract event."""
        if t > self.maturity + 1e-10:
            return 0.0, 0.0
        if side == "post" and abs(t - self.maturity) < 1e-10:
            return 0.0, 0.0
        if self.kind == "call":
            return 0.0, smax - self.strike
        if self.kind == "put":
            return self.strike, 0.0
        if self.kind == "spread":
            return 0.0, self.strike2 - self.strike
        if self.kind == "digital":
            return 0.0, 1.0
        if self.kind == "upout":
            return 0.0, 0.0
        if self.kind == "autocall":
            y = int(np.floor(t + 1e-10)) + 1 if side == "post" else int(np.ceil(t - 1e-10))
            y = max(1, min(y, self.maturity))
            return 0.0, self.redemption(y)
        raise ValueError(self.kind)


@dataclass
class Grid:
    """Time and spot axes with product values and post-event slices for interpolation."""

    t: np.ndarray
    s: np.ndarray
    values: np.ndarray  # [time, spot, product] or [time,spot,scenario,product]
    post_events: dict

    def sample(self, t, s, side="post"):
        """Bilinear interpolation; no interpolation across an autocall jump.

        The next interval starts from post-event, the preceding one ends at pre.
        Extrapolation of risk (not spot) outside the space grid is zero.
        """
        s = np.atleast_1d(np.asarray(s, dtype=float))
        pos = np.clip(t / (self.t[1] - self.t[0]), 0, len(self.t) - 1)
        k = int(np.floor(pos + 1e-10))
        k = min(k, len(self.t) - 1)
        a = max(0.0, pos - k)
        left = (
            self.post_events.get(k, self.values[k])
            if side == "post" or a > 1e-9
            else self.values[k]
        )
        if a > 1e-9 and k + 1 < len(self.t):
            value = (1 - a) * self._space(left, s) + a * self._space(self.values[k + 1], s)
        else:
            value = self._space(left, s)
        return value

    def _space(self, v, s):
        """Linearly interpolate one time slice along the spot axis."""
        p = np.clip(s / (self.s[1] - self.s[0]), 0, len(self.s) - 1)
        j = np.minimum(p.astype(int), len(self.s) - 2)
        a = (p - j).reshape((-1,) + (1,) * (v.ndim - 1))
        return (1 - a) * v[j] + a * v[j + 1]


def flat_sigma(level):
    """Return a volatility function with the same level at every time and spot."""
    return lambda t, s: np.full_like(np.asarray(s, dtype=float), level)


def deformation(base, eps, lam=0.0, nu=0.0):
    """Add eps * exp(-lam*t - nu*log(S/100)) to the base local volatility."""
    return lambda t, s: base(t, s) + eps * np.exp(
        -lam * t - nu * np.log(np.maximum(s, 1e-8) / 100)
    )


def proxy_sigma(t, s):
    """SYNTHETIC, uncalibrated local vol. Parameters are ours, not the paper's."""
    return 0.24 - 0.14 * np.tanh(1.5 * np.log(np.maximum(s, 1e-8) / 100)) + 0.02 * np.exp(-t)


def proxy_skew(t, s):
    """Apply the synthetic skew bump to the proxy local-volatility surface."""
    return proxy_sigma(t, s) - 0.01 * np.tanh(2 * np.log(np.maximum(s, 1e-8) / 100))


def proxy_model():
    """Build the optional synthetic proxy and its parallel and skew scenarios."""
    return {
        "base": proxy_sigma,
        "parallel": deformation(proxy_sigma, 0.01),
        "skew": proxy_skew,
        "tag": "proxy",
        "status": "synthetic_proxy_not_market_reproduction",
        "deformation_space": "local",
        "label": "SYNTHETIC PROXY - not the paper's market data",
        "metadata": {
            "model": "Original declared synthetic local-volatility proxy",
            "sigma": "0.24-0.14*tanh(1.5*ln(S/100))+0.02*exp(-t)",
            "synthetic": True,
        },
    }


class NormalizedDupire:
    """Quote-unit surface -> S0=100 research engine, with boundary counters."""

    def __init__(self, surface):
        """Store the quote surface and initialize counters for out-of-domain evaluations."""
        self.surface = surface
        self.counts = {
            "evaluated_points": 0,
            "outside_quote_strikes": 0,
            "before_first_expiry": 0,
        }

    def __call__(self, t, s):
        """Map engine spot units to quote units, count boundary visits and evaluate local volatility."""
        tt, ss = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(s, dtype=float))
        kk = np.log(np.maximum(ss, 1e-300) / 100)
        self.counts["evaluated_points"] += ss.size
        self.counts["outside_quote_strikes"] += int(
            np.sum((kk < self.surface.kmin) | (kk > self.surface.kmax))
        )
        self.counts["before_first_expiry"] += int(np.sum(tt < self.surface.tmin))
        return self.surface(t, ss * self.surface.spot / 100)


def build_dupire_market(
    out, *, vol_surface=None, surface_spot=100.0, deformation_space="local"
):
    """Build base and bumped Dupire models from supplied or synthetic implied quotes."""
    folder = Path(out) / "dupire"
    folder.mkdir(parents=True, exist_ok=True)
    if vol_surface is None:
        source = folder / "synthetic_dupire_surface.csv"
        provenance = write_demo_surface(source, spot=surface_spot)
        surface = DupireLocalVol.from_csv(source, spot=surface_spot)
        surface.source.update(provenance)
        tag = "dupire_demo"
        label = "DUPIRE ON A SYNTHETIC SURFACE - not the paper's Eurostoxx data"
        status = "dupire_calibrated_to_declared_synthetic_implied_surface"
    else:
        source = Path(vol_surface).expanduser().resolve()
        surface = DupireLocalVol.from_csv(source, spot=surface_spot)
        destination = folder / "input_implied_surface.csv"
        if source != destination.resolve():
            destination.write_bytes(source.read_bytes())
        tag = "dupire_input"
        label = "DUPIRE ON SUPPLIED CSV - provenance and conventions documented in the report"
        status = "dupire_from_user_csv_source_not_independently_verified"
    if surface.tmax < 5 - 1e-10:
        raise DupireSurfaceError(
            "The autocall and puts require quotes extending to at least 5 years; missing maturities are not fabricated."
        )
    if not surface.kmin < 0 < surface.kmax:
        raise DupireSurfaceError("The quote grid must cover strikes below and above spot.")
    base = NormalizedDupire(surface)
    model = {
        "base": base,
        "surface": surface,
        "tag": tag,
        "label": label,
        "status": status,
        "deformation_space": "local" if deformation_space == "local" else "implied",
        "metadata": surface.metadata(),
    }
    if deformation_space == "implied":
        parallel = surface.bumped(parallel=0.01)
        skew = surface.bumped(skew=0.01)
        model.update(parallel=NormalizedDupire(parallel), skew=NormalizedDupire(skew))
        model["metadata"]["bump_validation"] = {
            "parallel": parallel.validation,
            "skew": skew.validation,
        }
    else:
        model["parallel"] = deformation(base, 0.01)
        model["skew"] = lambda t, s: base(t, s) - 0.01 * np.tanh(
            2 * np.log(np.maximum(s, 1e-8) / 100)
        )
        # Positive local vol must hold on the same space/time region used by the
        # PDE, including the chosen wing continuation. Fail rather than clip.
        tt = np.linspace(0, 5, 101)[:, None]
        ss = np.r_[0, np.geomspace(0.001, 600, 301)][None, :]
        if np.any(model["skew"](tt, ss) <= 0):
            raise DupireSurfaceError(
                "The local skew shift produces nonpositive volatility; revise the scenario or surface."
            )
    model["metadata"]["deformation_space"] = model["deformation_space"]
    write_json(folder / "surface_metadata.json", model["metadata"])
    return model


def build_dupire_model(args, out):
    """Support earlier research scripts that pass model settings as a namespace."""
    return build_dupire_market(
        out,
        vol_surface=None if args.dupire_demo else args.vol_surface,
        surface_spot=args.surface_spot,
        deformation_space=args.deformation_space,
    )


def dupire_diagnostics(model, cfg, out, tolerance_vol_bp=5.0):
    """Independent limits plus backward-PDE repricing, with explicit coverage.

    One vol-bp=0.0001 in decimal volatility. Repricing covers selected liquid
    strikes and maturities; it is not a proof about all quotes or all strikes.
    """
    folder = Path(out) / "dupire"
    folder.mkdir(parents=True, exist_ok=True)
    log("[Dupire] Independent checks and vanilla repricing")
    unit = validate_dupire_fragment()
    surface = model["surface"]
    lower = max(70, 100 * np.exp(surface.kmin) + 1e-6)
    upper = min(160, 100 * np.exp(surface.kmax) - 1e-6)
    if lower >= upper:
        raise DupireSurfaceError("Insufficient strike coverage for the repricing check")
    # Align all strikes with the space mesh to avoid attributing an unrelated
    # off-grid payoff error to the extracted local volatility.
    strikes = np.unique(np.round(np.linspace(lower, upper, 9) / cfg.ds) * cfg.ds)
    strikes = strikes[(strikes >= lower) & (strikes <= upper)]
    rows = []
    invalid_iv = 0
    for expiry in [1, 2, 3, 5]:
        grid = solve_pde(
            [Product("call", expiry, float(k)) for k in strikes], model["base"], cfg
        )
        actual = grid.sample(0, [100])[0]
        inputvol = surface.implied_vol(expiry, strikes * surface.spot / 100)
        target = bs_price(100, strikes, expiry, inputvol)
        for k, price, p0, iv0 in zip(strikes, actual, target, inputvol):
            if not max(100 - k, 0) < price < 100:
                invalid_iv += 1
                iv = None
                error = None
            else:
                iv = brentq(lambda vol: float(bs_price(100, k, expiry, vol)) - price, 1e-5, 5.0)
                error = (iv - iv0) * 1e4
            rows.append(
                [expiry, k, float(iv0), float(p0), float(price), float(price - p0), iv, error]
            )
        del grid
    valid = [r[7] for r in rows if r[7] is not None]
    maxerr = max(map(abs, valid), default=float("inf"))
    result = {
        "unit_checks": unit,
        "model": model["metadata"],
        "quote_units": "decimal implied vol; rate=dividend=0",
        "repricing": {
            "tested_instruments": len(rows),
            "maturities_years": [1, 2, 3, 5],
            "normalized_strikes": strikes.tolist(),
            "max_abs_price_error_points": float(max(abs(r[5]) for r in rows)),
            "max_abs_implied_vol_error_bp": float(maxerr),
            "tolerance_implied_vol_bp": tolerance_vol_bp,
            "invalid_implied_vol_count": invalid_iv,
            "passed": bool(invalid_iv == 0 and maxerr <= tolerance_vol_bp),
            "coverage_note": "Selected strikes and maturities, not a check of every CSV quote or a global calibration certificate.",
        },
        "boundary_evaluations_before_fhc": dict(model["base"].counts),
    }
    write_csv(
        folder / "vanilla_repricing.csv",
        [
            "maturity",
            "strike_normalized",
            "input_implied_vol",
            "input_BS_price",
            "Dupire_PDE_price",
            "price_error",
            "reimplied_vol",
            "iv_error_bp",
        ],
        rows,
    )
    write_json(folder / "dupire_validation.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    ts = np.linspace(max(0.01, surface.tmin), 5, 81)
    spots = np.linspace(lower, upper, 100)
    tt, ss = np.meshgrid(ts, spots, indexing="ij")
    implied = surface.implied_vol(tt, ss * surface.spot / 100)
    local = model["base"](tt, ss)
    for ax, z, title in zip(
        axes,
        [implied, local],
        ["Input implied-volatility surface", "Local volatility extracted with Dupire"],
    ):
        p = ax.pcolormesh(spots, ts, 100 * z, shading="auto", cmap="viridis")
        ax.set(xlabel="Normalized strike / spot", ylabel="T / t (years)", title=title)
        fig.colorbar(p, ax=ax, label="Volatility (%)")
    fig_label(fig, "Dupire", "Implied and local volatility", model["label"])
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.35)
    savefig(fig, folder / "dupire_surfaces.png", cfg)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    for expiry in [1, 2, 3, 5]:
        selected = [r for r in rows if r[0] == expiry and r[7] is not None]
        ax.plot(
            [r[1] for r in selected], [r[7] for r in selected], "o-", label=f"T={expiry} years"
        )
    ax.axhline(0, color="grey", lw=0.8)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set(xlabel="Normalized strike (S0=100)", ylabel="Implied-volatility error (absolute bp)")
    fig_label(fig, "Dupire", "PDE repricing against the input surface", model["label"])
    fig.subplots_adjust(top=0.84, bottom=0.16)
    savefig(fig, folder / "dupire_repricing.png", cfg)
    log(f"[Dupire] Maximum repricing error: {maxerr:.4g} volatility bp")
    if not result["repricing"]["passed"]:
        raise DupireSurfaceError(
            "Repricing exceeds tolerance: see dupire/dupire_validation.json. Improve the surface, domain or mesh; do not disregard the discrepancy."
        )
    return result


def solve_pde(products, sigma, cfg, smax=None):
    """u_t + (sigma(t,S)^2 S^2/2)u_SS = 0, with event conditions.

    Multiple products share the tridiagonal factorization at each step. Price
    grids are stored in chronological order. Boundaries are Dirichlet; r=q=0.
    """
    smax = cfg.smax if smax is None else smax
    n = int(round(smax / cfg.ds))
    s = np.linspace(0, smax, n + 1)
    if not np.isclose(s[1], cfg.ds):
        raise ValueError("Smax must be a multiple of ds")
    T = max(p.maturity for p in products)
    nt = T * cfg.steps_year
    times = np.linspace(0, T, nt + 1)
    dt = 1 / cfg.steps_year
    values = np.zeros((nt + 1, n + 1, len(products)))
    for j, p in enumerate(products):
        if p.maturity == T:
            values[-1, :, j] = p.payoff(s)
            values[-1, 0, j], values[-1, -1, j] = p.boundaries(T, smax)
    current = values[-1].copy()
    post = {nt: np.zeros_like(current)}
    damp = True

    def smooth_terminal(v, p, j):
        """Average the payoff limits at autocall jumps before backward propagation."""
        if p.kind == "autocall":
            ib = int(round(p.barrier / cfg.ds))
            ia = int(round(100 / cfg.ds))
            if np.isclose(s[ib], p.barrier):
                v[ib, j] = 0.5 * (p.barrier / 100 + 1)
            if np.isclose(s[ia], 100):
                v[ia, j] = 0.5 * (1 + p.redemption(p.maturity))

    for j, p in enumerate(products):
        if p.maturity == T:
            smooth_terminal(current, p, j)

    def step(v, t0, t1, theta, end_side="post"):
        """Advance one backward PDE step using theta and the chosen event-side boundaries."""
        h = t1 - t0
        mid = (t0 + t1) / 2
        sig = np.asarray(sigma(mid, s[1:-1]))
        if np.any(~np.isfinite(sig)) or np.any(sig <= 0):
            raise ValueError("Nonpositive or nonfinite volatility")
        a = 0.5 * sig**2 * s[1:-1] ** 2 / cfg.ds**2
        result = np.empty_like(v)
        for j, p in enumerate(products):
            result[0, j], result[-1, j] = p.boundaries(t0, smax, end_side)
        rhs = v[1:-1] + (1 - theta) * h * a[:, None] * (v[:-2] - 2 * v[1:-1] + v[2:])
        rhs[0] += theta * h * a[0] * result[0]
        rhs[-1] += theta * h * a[-1] * result[-1]
        ab = np.zeros((3, n - 1))
        ab[1] = 1 + 2 * theta * h * a
        ab[0, 1:] = -theta * h * a[:-1]
        ab[2, :-1] = -theta * h * a[1:]
        result[1:-1] = solve_banded((1, 1), ab, rhs, check_finite=False)
        return result

    for k in range(nt - 1, -1, -1):
        t0, t1 = times[k], times[k + 1]
        if damp:
            current = step(current, (t0 + t1) / 2, t1, 1.0)
            current = step(current, t0, (t0 + t1) / 2, 1.0)
        else:
            current = step(current, t0, t1, 0.5)
        damp = False
        event = k > 0 and k % cfg.steps_year == 0
        if event:
            post[k] = current.copy()
            year = k // cfg.steps_year
            for j, p in enumerate(products):
                if p.maturity == year:
                    current[:, j] = p.payoff(s)
                    damp = True
                elif p.kind == "autocall" and year < p.maturity:
                    current[s >= 100, j] = p.redemption(year)
                    damp = True
        for j, p in enumerate(products):
            if times[k] > p.maturity + 1e-9:
                current[:, j] = 0
            current[0, j], current[-1, j] = p.boundaries(times[k], smax, "pre")
        values[k] = current
        # Smooth only the initial data for backward propagation. Exact pre and
        # post event states stored above remain available to interpolation/MC.
        if event:
            current = current.copy()
            year = k // cfg.steps_year
            ia = int(round(100 / cfg.ds))
            for j, p in enumerate(products):
                if p.maturity == year:
                    smooth_terminal(current, p, j)
                elif p.kind == "autocall" and year < p.maturity and np.isclose(s[ia], 100):
                    current[ia, j] = 0.5 * (post[k][ia, j] + p.redemption(year))
    return Grid(times, s, values, post)


def risk_bundle(products, sigmas, cfg, solver=None):
    """Finite price differences; divide by eps only when reporting a vega."""
    solver = solver or (lambda sig: solve_pde(products, sig, cfg))
    base = solver(sigmas[0])
    shocks = []
    post = {k: [] for k in base.post_events}
    for sig in sigmas[1:]:
        bumped = solver(sig)
        shocks.append(bumped.values - base.values)
        for k in post:
            post[k].append(bumped.post_events[k] - base.post_events[k])
        del bumped
    delta = np.stack(shocks, axis=2)
    return base, Grid(base.t, base.s, delta, {k: np.stack(v, axis=1) for k, v in post.items()})


def barrier_solver(sigma, cfg):
    """Price the up-and-in via in-out parity and return its grid with the hedging call."""
    vanilla = solve_pde([Product("call", 2, 100), Product("call", 2, 120)], sigma, cfg)
    out = solve_pde([Product("upout", 2, 100)], sigma, cfg, smax=120)
    ui = vanilla.values[:, :, 0].copy()
    ui[:, : len(out.s)] -= out.values[:, :, 0]
    vals = np.stack([ui, vanilla.values[:, :, 1]], axis=-1)
    return Grid(vanilla.t, vanilla.s, vals, {})


def bs_price(s, k, tau, sigma, kind="call"):
    """Return the zero-rate Black–Scholes price for a call, put or cash digital."""
    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    v = np.asarray(sigma) * np.sqrt(np.maximum(tau, 1e-30))
    d1 = np.log(np.maximum(s, 1e-300) / k) / v + 0.5 * v
    d2 = d1 - v
    if kind == "call":
        result = s * ndtr(d1) - k * ndtr(d2)
        terminal = np.maximum(s - k, 0)
    elif kind == "put":
        result = k * ndtr(-d2) - s * ndtr(-d1)
        terminal = np.maximum(k - s, 0)
    elif kind == "digital":
        result = ndtr(d2)
        terminal = (s > k).astype(float) + 0.5 * (s == k)
    else:
        raise ValueError(kind)
    return np.where(tau <= 0, terminal, result)


def solve_weights(risk, cfg):
    """Small square systems via SVD; masks are reported, never filled with theory."""
    A = risk.values[:, :, :, 1:]
    b = risk.values[:, :, :, 0]
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    largest = s[..., 0]
    smallest = s[..., -1]
    identifiable = (smallest > cfg.min_scenario_price) & (smallest > cfg.rcond * largest)
    inv = np.divide(1, s, out=np.zeros_like(s), where=s > cfg.rcond * largest[..., None])
    w = np.einsum("...ji,...j->...i", Vh, inv * np.einsum("...ji,...j->...i", U, b))
    # U^T b and V (=Vh^T) above, with arrays ending in matrix dimensions.
    w[~identifiable] = np.nan
    residual = np.einsum("...ij,...j->...i", A, w) - b
    cond = np.divide(largest, smallest, out=np.full_like(largest, np.inf), where=smallest > 0)
    validcond = cond[identifiable]
    stats = {
        "total_nodes": int(identifiable.size),
        "identifiable_nodes": int(identifiable.sum()),
        "masked_nodes": int((~identifiable).sum()),
        "minimum_retained_singular_value_absolute": cfg.min_scenario_price,
        "condition_median": float(np.median(validcond)) if validcond.size else None,
        "condition_max": float(np.max(validcond)) if validcond.size else None,
        "max_scenario_residual": (
            float(np.nanmax(np.abs(residual))) if np.any(identifiable) else None
        ),
    }
    return w, cond, stats


def minimum_change(A, b, previous, rcond=1e-10, tolerance=1e-5):
    """Algorithm2 / Remark4.4. Shapes: (paths,risks,instruments)."""
    residual = b - np.einsum("nij,nj->ni", A, previous)
    do = np.max(np.abs(residual), axis=1) > tolerance
    delta = np.zeros_like(previous)
    if np.any(do):
        aa = A[do]
        rr = residual[do]
        if aa.shape[1] == 1:
            den = np.sum(aa[:, 0] ** 2, axis=1)
            scale = np.divide(rr[:, 0], den, out=np.zeros_like(den), where=den > 1e-30)
            delta[do] = scale[:, None] * aa[:, 0]
        else:
            # NumPy's batched SVD supplies the Moore-Penrose inverse, also at rank loss.
            pinv = np.linalg.pinv(aa, rcond=rcond)
            delta[do] = np.einsum("nji,ni->nj", pinv, rr)
    return previous + delta, delta, do


def savefig(fig, path, cfg):
    """Save a figure at the configured resolution, close it and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_reproduced_figure(fig, product, filename, cfg, results_dir="results"):
    """Save a recalculated figure in results/<product>/<filename>.png.

    Create the product folder as needed, save the Matplotlib figure as PNG
    at cfg.dpi, close the figure and return its path. Original images
    extracted from the PDF remain in original_figures.
    """
    path = Path(results_dir) / product / Path(filename).with_suffix(".png")
    return savefig(fig, path, cfg)


def surface(ax, t, s, z, title, zlabel="Weight", bounds=None):
    """Plot a time-by-spot surface using a subsample for rendering."""
    it = np.unique(np.linspace(0, len(t) - 1, min(56, len(t))).astype(int))
    js = np.unique(np.linspace(0, len(s) - 1, min(90, len(s))).astype(int))
    tt, ss = np.meshgrid(t[it], s[js], indexing="ij")
    zz = z[np.ix_(it, js)]
    if np.any(np.isfinite(zz)):
        ax.plot_surface(
            tt, ss, zz, cmap="viridis", linewidth=0, antialiased=True, rasterized=True
        )
    ax.set(xlabel="t (years)", ylabel="S", zlabel="", title=title)
    ax.text2D(0.02, 0.88, zlabel, transform=ax.transAxes, fontsize=9, color="#405269")
    ax.view_init(elev=27, azim=-125)
    if bounds is not None:
        ax.set_zlim(*bounds)


def subset(grid, z, tmax, smin, smax):
    """Select a time and spot window from a grid and its associated values."""
    it = grid.t < tmax - 1e-9
    js = (grid.s >= smin) & (grid.s <= smax)
    return grid.t[it], grid.s[js], z[it][:, js]


def fig_label(fig, number, title, proxy=False):
    """Add the paper figure number, title and model label to a reproduced figure."""
    fig.suptitle(f"Figure {number} | {title}", fontsize=13, fontweight="bold", y=0.99)
    text = (
        proxy
        if isinstance(proxy, str)
        else (
            "UNCALIBRATED SUBSTITUTE SCENARIO - not the paper's market data"
            if proxy
            else "PDE recalculation with declared conventions - not an exact numerical copy of the original figure"
        )
    )
    fig.text(0.5, 0.012, text, ha="center", fontsize=8, color="#9b3a24" if proxy else "#555555")


def robust_residual(weights, base, bumped, min_change=1e-9):
    """Express the hedge error as a percentage of the absolute exotic price change."""
    du = bumped.values[:, :, 0] - base.values[:, :, 0]
    dh = bumped.values[:, :, 1:] - base.values[:, :, 1:]
    numerator = np.sum(weights * dh, axis=-1) - du
    return np.divide(
        100 * numerator, np.abs(du), out=np.full_like(du, np.nan), where=np.abs(du) > min_change
    )


def section3(cfg, out, results_dir=None):
    """Reproduce the Section 3 examples and save figures, numerical grids and diagnostics."""
    results_dir = out.parent if results_dir is None else Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    data.mkdir(exist_ok=True)
    diag = {}
    basevol = flat_sigma(0.30)
    eps = 0.003
    log("[Section 3] Call spread and digitals: PDE and forward bumps")
    prods = [
        Product("spread", 2),
        Product("call", 2, 100),
        Product("call", 2, 110),
        Product("digital", 2, 100),
        Product("call", 2, 99),
    ]
    base, risk = risk_bundle(
        prods, [basevol, deformation(basevol, eps), deformation(basevol, eps, 1)], cfg
    )

    def select(indices, scens=None):
        """Extract chosen products and bump scenarios from the shared risk grid."""
        v = risk.values[:, :, :, indices]
        if scens is not None:
            v = v[:, :, scens, :]
        return Grid(risk.t, risk.s, v, {})

    spreadrisk = select([0, 1, 2])
    w1, c1, diag["fig01"] = solve_weights(spreadrisk, cfg)
    fig = plt.figure(figsize=(12, 5.3))
    for j, k in enumerate([100, 110]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        t, s, z = subset(base, w1[:, :, j], 2, 0, 200)
        surface(ax, t, s, z, f"Call K={k}, T=2", bounds=(0.90, 1.1) if j == 0 else (-1.1, -0.9))
    fig_label(fig, 1, "Call spread hedge")
    fig.subplots_adjust(top=0.83, bottom=0.12, wspace=0.3)
    fig.text(
        0.5,
        0.06,
        "Unidentifiable nodes masked: minimum singular value >1e-8; configurable relative SVD cutoff",
        ha="center",
        fontsize=8,
    )
    save_reproduced_figure(fig, "call_spread", "fig_01_recomputed.png", cfg, results_dir)
    check = (base.s >= 60) & (base.s <= 160)
    v = w1[: cfg.steps_year, check]
    diag["call_spread_weight_max_abs_error_central"] = float(np.nanmax(np.abs(v - [1, -1])))

    wrisk = select([3, 1], [0])
    w2, c2, diag["fig02"] = solve_weights(wrisk, cfg)
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_subplot(121)
    for t in [0, 1, 1.9]:
        mask = base.s <= 200
        ax.plot(
            base.s[mask],
            risk.sample(t, base.s[mask], side="pre")[:, 0, 3] / eps,
            label=f"t={t:g}",
        )
    ax.set(xlabel="S", ylabel="Vega per unit of decimal volatility", title="Digital K=100, T=2")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(122, projection="3d")
    surface(ax, *subset(base, w2[:, :, 0], 2, 0, 200), "Call weight K=100, T=2")
    fig_label(fig, 2, "Digital: one hedging instrument")
    fig.subplots_adjust(top=0.83, bottom=0.15, wspace=0.35)
    save_reproduced_figure(fig, "digital", "fig_02_recomputed.png", cfg, results_dir)
    wrisk = select([3, 4, 1])
    w3, c3, diag["fig03"] = solve_weights(wrisk, cfg)
    fig = plt.figure(figsize=(12, 10))
    for j in range(2):
        for q, (lo, hi) in enumerate([(0, 99.5), (100.5, 200)]):
            ax = fig.add_subplot(2, 2, 2 * j + q + 1, projection="3d")
            surface(
                ax,
                *subset(base, w3[:, :, j], 2, lo, hi),
                f"Call K={[99,100][j]} | {'S<100' if q==0 else 'S>100'}",
            )
    fig_label(fig, 3, "Digital: two time-dependent bumps, sensitive system")
    fig.subplots_adjust(top=0.91, bottom=0.08, wspace=0.26, hspace=0.30)
    fig.text(
        0.5,
        0.045,
        "Unidentifiable nodes masked; complete grids and condition numbers saved in NPZ data",
        ha="center",
        fontsize=8,
    )
    save_reproduced_figure(fig, "digital", "fig_03_recomputed.png", cfg, results_dir)
    np.savez_compressed(
        data / "fig01_03_grids.npz",
        time=base.t,
        spot=base.s,
        prices=base.values,
        changes=risk.values,
        weights_spread=w1,
        weights_digital_one=w2,
        weights_digital_two=w3,
        condition_spread=c1,
        condition_digital=c3,
    )
    del base, risk, spreadrisk, wrisk, w1, w2, w3, c1, c2, c3
    gc.collect()

    log("[Section 3] Up-and-in: continuous barrier and in/out parity")
    solver = lambda sig: barrier_solver(sig, cfg)
    base, risk = risk_bundle([], [basevol, deformation(basevol, eps)], cfg, solver)
    w, cond, diag["fig04"] = solve_weights(risk, cfg)
    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_subplot(121)
    for t in [0, 1, 1.9]:
        mask = base.s <= 140
        ax.plot(
            base.s[mask],
            risk.sample(t, base.s[mask], side="pre")[:, 0, 0] / eps,
            label=f"t={t:g}",
        )
    ax.set(xlabel="S", ylabel="Vega", title="Up-and-in K=100, B=120, T=2")
    ax.legend()
    ax.grid(alpha=0.25)
    ax = fig.add_subplot(122, projection="3d")
    surface(ax, *subset(base, w[:, :, 0], 2, 0, 120), "Call weight K=120")
    fig_label(fig, 4, "Up-and-in call")
    fig.subplots_adjust(top=0.83, bottom=0.15, wspace=0.35)
    save_reproduced_figure(fig, "up_and_in_call", "fig_04_recomputed.png", cfg, results_dir)
    np.savez_compressed(
        data / "fig04_grids.npz",
        time=base.t,
        spot=base.s,
        prices=base.values,
        changes=risk.values,
        weights=w,
    )
    del base, risk, w, cond
    gc.collect()

    sig20 = flat_sigma(0.20)
    e20 = 0.002
    base, risk = risk_bundle([], [sig20, deformation(sig20, e20)], cfg, solver)
    w, _, _ = solve_weights(risk, cfg)
    fig = plt.figure(figsize=(12, 10))
    resid = []
    raw_resid = []
    pricechanges = []
    specs = [(-e20, 0, 0), (2 * e20, 0, 0), (e20, 1, 0), (e20, 1, 0.01)]
    for k, (amp, lam, nu) in enumerate(specs):
        bumped = solver(deformation(sig20, amp, lam, nu))
        raw = robust_residual(w, base, bumped, min_change=1e-12)
        du = bumped.values[:, :, 0] - base.values[:, :, 0]
        rr = raw.copy()
        rr[np.abs(du) < 1e-6] = np.nan
        resid.append(rr)
        raw_resid.append(raw)
        pricechanges.append(du)
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        surface(
            ax,
            *subset(base, rr, 2, 1, 119),
            f"bump={amp:g}; lambda={lam}; nu={nu}",
            "Residual (%)",
        )
    fig_label(fig, 5, "Up-and-in robustness, sigma=20%")
    fig.subplots_adjust(top=0.91, bottom=0.08, wspace=0.28, hspace=0.3)
    fig.text(
        0.5,
        0.045,
        "Ratios with |price change| <1e-6 masked; unfiltered ratios saved in the data",
        ha="center",
        fontsize=8,
    )
    save_reproduced_figure(fig, "up_and_in_call", "fig_05_recomputed.png", cfg, results_dir)
    np.savez_compressed(
        data / "fig05_grids.npz",
        time=base.t,
        spot=base.s,
        weights=w,
        residuals=np.array(resid),
        residuals_raw=np.array(raw_resid),
        price_changes=np.array(pricechanges),
    )
    del base, risk, w, rr, bumped, resid, raw_resid, pricechanges
    gc.collect()

    log("[Section 3] Three-year autocall and robustness")
    prods = [Product("autocall", 3), Product("call", 2, 100), Product("call", 3, 100)]
    base, risk = risk_bundle(
        prods, [basevol, deformation(basevol, eps), deformation(basevol, eps, 0.5)], cfg
    )
    w, cond, diag["fig08"] = solve_weights(risk, cfg)
    fig, ax = plt.subplots(figsize=(8.5, 5.7))
    mask = base.s <= 300
    for t in [0, 0.5, 0.9]:
        v = risk.sample(t, base.s[mask], side="pre")[:, 0, 0] / eps
        ax.plot(base.s[mask], v, label=f"t={t:g}")
    ax.set(xlabel="S/S0 (%)", ylabel="Vega, unit notional", title="Year 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_label(fig, 7, "Three-year autocall vega")
    fig.subplots_adjust(top=0.82, bottom=0.15)
    save_reproduced_figure(fig, "autocall_3_years", "fig_07_recomputed.png", cfg, results_dir)
    fig = plt.figure(figsize=(12, 5.4))
    for j in range(2):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        surface(ax, *subset(base, w[:, :, j], 1, 40, 200), f"Call K=100, T={j+2}")
    fig_label(fig, 8, "Autocall hedge: two maturities")
    fig.subplots_adjust(top=0.83, bottom=0.15, wspace=0.3)
    save_reproduced_figure(fig, "autocall_3_years", "fig_08_recomputed.png", cfg, results_dir)
    fig = plt.figure(figsize=(12, 10))
    resid = []
    raw_resid = []
    pricechanges = []
    specs = [(-eps, 0), (2 * eps, 0), (eps, 0.25), (eps, 0.75)]
    for k, (amp, lam) in enumerate(specs):
        bumped = solve_pde(prods, deformation(basevol, amp, lam), cfg)
        raw = robust_residual(w, base, bumped, min_change=1e-12)
        du = bumped.values[:, :, 0] - base.values[:, :, 0]
        rr = raw.copy()
        rr[np.abs(du) < 1e-5] = np.nan
        resid.append(rr)
        raw_resid.append(raw)
        pricechanges.append(du)
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        surface(
            ax, *subset(base, rr, 1, 40, 200), f"bump={amp:g}; lambda={lam}", "Residual (%)"
        )
    fig_label(fig, 9, "Autocall hedge robustness")
    fig.subplots_adjust(top=0.91, bottom=0.08, wspace=0.28, hspace=0.3)
    fig.text(
        0.5,
        0.045,
        "Ratios with |price change| <0.1 bp masked; unfiltered ratios saved in the data",
        ha="center",
        fontsize=8,
    )
    save_reproduced_figure(fig, "autocall_3_years", "fig_09_recomputed.png", cfg, results_dir)
    fig = plt.figure(figsize=(12, 10))
    for k, ((amp, lam), rr) in enumerate(zip(specs, resid)):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        surface(
            ax, *subset(base, rr, 1, 70, 150), f"bump={amp:g}; lambda={lam}", "Residual (%)"
        )
    fig_label(fig, "9 - detail", "Declared zoom S=70..150, near the autocall level")
    fig.subplots_adjust(top=0.91, bottom=0.08, wspace=0.28, hspace=0.3)
    fig.text(
        0.5,
        0.045,
        "Supplementary chart: main Figure 9 retains the S=40..200 domain and peaks in the tails",
        ha="center",
        fontsize=8,
    )
    save_reproduced_figure(
        fig, "autocall_3_years", "fig_09_zoom_supplementary.png", cfg, results_dir
    )
    np.savez_compressed(
        data / "fig07_09_grids.npz",
        time=base.t,
        spot=base.s,
        prices=base.values,
        changes=risk.values,
        weights=w,
        conditions=cond,
        residuals=np.array(resid),
        residuals_raw=np.array(raw_resid),
        price_changes=np.array(pricechanges),
    )
    diag["autocall_price_S100_t0"] = float(base.sample(0, [100])[0, 0])
    diag["autocall_parallel_vega_S100_t0"] = float(risk.sample(0, [100])[0, 0, 0] / eps)
    diag["residual_ratio_display_thresholds"] = {
        "figure5": 1e-6,
        "figure9": 1e-5,
        "raw_ratio_floor": 1e-12,
    }
    write_json(out / "diagnostics.json", diag)
    return diag


def open_export_csv(path, mode="wt"):
    """Open UTF-8 CSV, optionally gzip-compressed without changing values."""
    path = Path(path)
    if "w" in mode:
        path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def export_csv_path(folder, name, cfg):
    """Build a CSV output path with the configured compression suffix."""
    return Path(folder) / (name + (".gz" if cfg.csv_compression == "gzip" else ""))


def write_numeric_export(path, columns, blocks):
    """Write blocks with 17 significant digits, without subsampling."""
    rows = 0
    with open_export_csv(path) as handle:
        handle.write(",".join(columns) + "\n")
        for block in blocks:
            block = np.asarray(block)
            if not block.size:
                continue
            np.savetxt(handle, block, delimiter=",", fmt="%.17g")
            rows += len(block)
    return rows


def export_algorithm2_data(base, risk, products, model, cfg, folder):
    """Export the actual PDE/MC arrays, coordinates and conventions.

    Sensitivities are finite PRICE differences, not normalized derivatives.
    The first product column is b (autocall), and the others form A
    (puts K=30,...,160). Post-event states are stored separately.
    CsvMcTrace writes path-dependent weights during Monte Carlo simulation.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ids = ["autocall" if p.kind == "autocall" else f"{p.kind}_K{p.strike:g}" for p in products]
    if len(set(ids)) != len(ids):
        raise ValueError("Instrument identifiers must be unique")
    events = sorted(base.post_events)
    if events != sorted(risk.post_events):
        raise ValueError("PDE and risk events are not aligned")
    header = ["time_index", "spot_index", "time_years", "spot"] + ids
    files = {}
    counts = {}
    write_csv(
        folder / "instruments.csv",
        [
            "product_index",
            "instrument_id",
            "kind",
            "maturity_years",
            "strike",
            "barrier",
            "coupon",
            "coupon_mode",
            "price_unit",
        ],
        (
            [
                j,
                ids[j],
                p.kind,
                p.maturity,
                p.strike,
                p.barrier,
                p.coupon,
                p.coupon_mode,
                "per_unit_note_notional" if j == 0 else "index_points",
            ]
            for j, p in enumerate(products)
        ),
    )
    write_csv(
        folder / "pde_time_axis.csv",
        ["time_index", "time_years", "has_post_event_state"],
        ([i, float(t), int(i in base.post_events)] for i, t in enumerate(base.t)),
    )
    write_csv(folder / "pde_spot_axis.csv", ["spot_index", "spot"], enumerate(base.s))

    def grid_blocks(values, post=None):
        """Yield one spot slice per time index, including post-event slices when requested."""
        indices = range(len(base.t)) if post is None else events
        for i in indices:
            value = values[i] if post is None else post[i]
            yield np.column_stack(
                (
                    np.full(len(base.s), i),
                    np.arange(len(base.s)),
                    np.full(len(base.s), base.t[i]),
                    base.s,
                    value,
                )
            )

    arrays = [
        ("prices_base", "pde_base_prices.csv", base.values, base.post_events),
        (
            "risk_parallel",
            "put_parallel_sensitivities.csv",
            risk.values[:, :, 0, :],
            {i: v[:, 0, :] for i, v in risk.post_events.items()},
        ),
        (
            "risk_skew",
            "put_skew_sensitivities.csv",
            risk.values[:, :, 1, :],
            {i: v[:, 1, :] for i, v in risk.post_events.items()},
        ),
    ]
    for key, name, values, post in arrays:
        log(f"[Algorithm 2 CSV] {name}")
        path = export_csv_path(folder, name, cfg)
        files[key] = path.name
        counts[key] = write_numeric_export(path, header, grid_blocks(values))
        path = export_csv_path(folder, name.replace(".csv", "_post_event.csv"), cfg)
        files[key + "_post"] = path.name
        counts[key + "_post"] = write_numeric_export(path, header, grid_blocks(None, post))

    # Include both PDE sampling times (midpoints/quarter-points under Rannacher)
    # and MC evaluation times. Spatial values remain a tabulation, not a new
    # interpolant substituted into the pricing or Monte Carlo model.
    coefficient_times = []
    for i in range(len(base.t) - 1):
        a, b = base.t[i : i + 2]
        damp = (i + 1 == len(base.t) - 1) or ((i + 1) % cfg.steps_year == 0)
        if damp:
            mid = (a + b) / 2
            coefficient_times.extend(((a + mid) / 2, (mid + b) / 2))
        else:
            coefficient_times.append((a + b) / 2)
    mc_times = np.arange(1260, dtype=float) / 252
    vol_times = np.unique(np.r_[base.t, coefficient_times, mc_times])
    nodes = set(map(float, base.t))
    coeff = set(map(float, coefficient_times))
    mc = set(map(float, mc_times))
    write_csv(
        folder / "volatility_surface_times.csv",
        [
            "vol_time_index",
            "time_years",
            "pde_node",
            "pde_coefficient_time",
            "mc_evaluation_time",
        ],
        (
            [i, float(t), int(t in nodes), int(t in coeff), int(t in mc)]
            for i, t in enumerate(vol_times)
        ),
    )
    tracked = {
        id(fn): fn
        for fn in (model["base"], model["parallel"], model["skew"])
        if isinstance(fn, NormalizedDupire)
    }
    saved_counts = {key: dict(fn.counts) for key, fn in tracked.items()}

    def volatility_blocks():
        """Yield spot slices of the base and bumped local-volatility surfaces."""
        for i, t in enumerate(vol_times):
            values = [
                np.broadcast_to(fn(float(t), base.s), base.s.shape)
                for fn in (model["base"], model["parallel"], model["skew"])
            ]
            yield np.column_stack(
                (
                    np.full(len(base.s), i),
                    np.arange(len(base.s)),
                    np.full(len(base.s), t),
                    base.s,
                    *values,
                )
            )

    path = export_csv_path(folder, "local_volatility_surfaces.csv", cfg)
    files["local_volatility"] = path.name
    try:
        log("[Algorithm 2 CSV] Base, parallel and skew local-volatility surfaces")
        counts["local_volatility"] = write_numeric_export(
            path,
            [
                "vol_time_index",
                "spot_index",
                "time_years",
                "spot",
                "sigma_base",
                "sigma_parallel",
                "sigma_skew",
            ],
            volatility_blocks(),
        )
    finally:
        for key, fn in tracked.items():
            fn.counts.clear()
            fn.counts.update(saved_counts[key])

    implied_files = {}
    implied_sources = {}
    # Quote CSVs are small and deliberately remain plain CSV so they can be
    # supplied directly to vol_surface or DupireLocalVol.from_csv.
    for key, label in [("base", "base"), ("parallel", "parallel"), ("skew", "skew")]:
        fn = model[key]
        if not isinstance(fn, NormalizedDupire):
            continue
        surface = fn.surface
        strikes = surface.spot * np.exp(surface.log_moneyness)
        path = folder / f"implied_volatility_{label}.csv"
        write_csv(
            path,
            ["maturity", "strike", "implied_vol", "log_moneyness"],
            (
                [
                    float(t),
                    float(k),
                    float(surface.implied_vols[i, j]),
                    float(surface.log_moneyness[j]),
                ]
                for i, t in enumerate(surface.maturities)
                for j, k in enumerate(strikes)
            ),
        )
        implied_files[key] = path.name
        implied_sources[key] = dict(surface.source)

    initial = risk.sample(0, [100], side="post")[0]
    initial_rows = []
    for nfactors in (1, 2):
        w = np.linalg.pinv(initial[:nfactors, 1:], rcond=cfg.rcond) @ initial[:nfactors, 0]
        initial_rows.extend(
            [nfactors, 0.0, 100.0, j, ids[j + 1], products[j + 1].strike, float(weight)]
            for j, weight in enumerate(w)
        )
    write_csv(
        folder / "initial_put_weights.csv",
        ["factors", "time_years", "spot", "hedge_column", "instrument_id", "strike", "weight"],
        initial_rows,
    )
    metadata = {
        "format_version": 1,
        "model_tag": model["tag"],
        "model_status": model["status"],
        "model_label": model["label"],
        "configuration": asdict(cfg),
        "volatility_model": model["metadata"],
        "quoted_spot": float(model["surface"].spot) if "surface" in model else 100.0,
        "engine_initial_spot": 100.0,
        "deformation_space": model["deformation_space"],
        "products": [
            {"index": j, "instrument_id": ids[j], "parameters": asdict(p)}
            for j, p in enumerate(products)
        ],
        "grid_shape_prices": list(base.values.shape),
        "grid_shape_risk": list(risk.values.shape),
        "post_event_time_indices": events,
        "pde_csv_columns": header,
        "files": files,
        "row_counts": counts,
        "implied_surface_files": implied_files,
        "implied_surface_sources": implied_sources,
        "scenario_order": ["parallel", "skew"],
        "bump_amplitude_decimal": 0.01,
        "matrix_A_columns": ids[1:],
        "target_b_column": ids[0],
        "sensitivities": "Raw finite price differences: deformed price minus base price. A=risk[...,1:]; b=risk[...,0].",
        "vega_units": "Divide the parallel price difference by 0.01 for the decimal-vol vega used in transaction costs.",
        "weights_units": "Number of puts per unit notional of the autocall; not percentage portfolio weights.",
        "price_units": "Autocall per unit note notional; put prices in normalized index points, S0=100.",
        "event_states": "Main arrays are pre-event at annual dates; separate files contain post-event continuation, including zero at T=5.",
        "interpolation": "Use Grid.sample: post of left node and pre of right node; explicit side at exact event dates.",
        "outside_pde_domain": "MC sets all risk changes to zero for S>grid.s[-1], after Grid.sample; preserve this rule.",
        "volatility_sampling": "All PDE nodes, PDE coefficient evaluation times and MC dates, at PDE spot nodes. These CSVs are samples of the model, not a replacement interpolant for off-grid spot values.",
        "implied_surface_scope": "Original input IV quotes; deformed IV quotes also exported when bumps are on implied vol. Local-vol bumps do not have input implied quotes.",
        "float_format": "CSV numeric grids/traces use 17 significant digits; no grid or path subsampling.",
        "mc_trace_scope": cfg.trace_runs,
        "mc_trace_policy": "All paths of both principal simulations by default. trace_runs='all' also exports pilot, profiles and hedging-frequency runs. Every run is listed in mc_runs.csv.",
        "mc_trace_sparse": "Initial state, every attempted correction and each termination. Carry weights forward by path between rows; omission means unchanged weights, not an omitted path.",
        "mc_event_codes": {
            "0": "initial hedge, cost excluded",
            "1": "hedge correction",
            "2": "autocall with hedge unwind",
            "3": "natural expiry, zero economic positions without transaction cost",
            "4": "autocall without unwind, tracking stops and final stored weights are retained",
        },
        "mc_delta_weights": "dw is the correction used by the algorithm; at initial hedge or expiry it describes a position change with zero charged turnover.",
    }
    write_json(folder / "algorithm2_metadata.json", metadata)
    return metadata


def save_results_csv(base, risk, products, model, cfg, results_dir=None):
    """Save grids, surfaces and sensitivities as CSV in the results directory.

    The default directory is next to this script, independently of the
    terminal or IDE working directory. Create the directory and prepare to
    save the actual hedge weights during Monte Carlo simulation.
    section4 calls this function automatically; no manual invocation is needed.
    """
    folder = (
        Path(__file__).resolve().parent / "results"
        if results_dir is None
        else Path(results_dir).expanduser().resolve()
    )
    folder.mkdir(parents=True, exist_ok=True)
    metadata = export_algorithm2_data(base, risk, products, model, cfg, folder)
    log(f"[CSV] Grids, surfaces and sensitivities saved in: {folder}")
    return Algorithm2Exports(folder, metadata, cfg, model)


def load_algorithm2_data(folder):
    """Reload prices, risks, events and the model without solving PDEs again.

    Example:
        data = load_algorithm2_data("results")
        risk = data["risk"]
        A = risk.sample(0.5, [100.0])[0, :, 1:]
        b = risk.sample(0.5, [100.0])[0, :, 0]
        simulation = simulate_fhc(risk, data["config"], npaths=100,
                                  sigma=data["model"]["base"])
    """
    folder = Path(folder)
    meta = json.loads((folder / "algorithm2_metadata.json").read_text(encoding="utf-8"))
    if meta.get("format_version") != 1:
        raise ValueError("Unsupported CSV format version")
    t_axis = np.loadtxt(folder / "pde_time_axis.csv", delimiter=",", skiprows=1, ndmin=2)
    s_axis = np.loadtxt(folder / "pde_spot_axis.csv", delimiter=",", skiprows=1, ndmin=2)
    t = t_axis[:, 1]
    s = s_axis[:, 1]
    nprod = len(meta["products"])
    if not (
        np.array_equal(t_axis[:, 0], np.arange(len(t)))
        and np.array_equal(s_axis[:, 0], np.arange(len(s)))
    ):
        raise ValueError("Invalid CSV axis indices")
    if np.any(np.diff(t) <= 0) or np.any(np.diff(s) <= 0):
        raise ValueError("CSV axes must be strictly increasing")
    if [len(t), len(s), nprod] != meta["grid_shape_prices"]:
        raise ValueError("CSV shape does not match metadata")
    events = meta["post_event_time_indices"]

    def read_grid(key, post=False):
        """Read an exported numerical grid and recover its declared axes and layout."""
        indices = events if post else range(len(t))
        values = {} if post else np.empty((len(t), len(s), nprod))
        with open_export_csv(folder / meta["files"][key], "rt") as handle:
            if handle.readline().rstrip("\r\n").split(",") != meta["pde_csv_columns"]:
                raise ValueError(f"Invalid columns: {key}")
            for i in indices:
                block = np.loadtxt(handle, delimiter=",", max_rows=len(s), ndmin=2)
                if (
                    block.shape != (len(s), 4 + nprod)
                    or not np.all(block[:, 0] == i)
                    or not np.array_equal(block[:, 1], np.arange(len(s)))
                    or not np.all(block[:, 2] == t[i])
                    or not np.array_equal(block[:, 3], s)
                ):
                    raise ValueError(f"Incomplete or reordered CSV grid: {key}, index {i}")
                values[i] = block[:, 4:]
            if handle.read().strip():
                raise ValueError(f"Unexpected rows: {key}")
        return values

    base = Grid(t, s, read_grid("prices_base"), read_grid("prices_base_post", post=True))
    delta = np.empty((len(t), len(s), 2, nprod))
    post = {i: np.empty((len(s), 2, nprod)) for i in events}
    for j, key in enumerate(("risk_parallel", "risk_skew")):
        delta[:, :, j, :] = read_grid(key)
        for i, values in read_grid(key + "_post", post=True).items():
            post[i][:, j, :] = values
    risk = Grid(t, s, delta, post)
    cfg = Config(**meta["configuration"])
    if meta["model_tag"] == "proxy":
        model = proxy_model()
    else:

        def read_implied(key):
            """Validate and reconstruct the rectangular implied-volatility quote grid."""
            path = folder / meta["implied_surface_files"][key]
            quotes = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            ts = np.unique(quotes[:, 0])
            ks = np.unique(quotes[:, 3])
            if (
                len(quotes) != len(ts) * len(ks)
                or not np.array_equal(quotes[:, 0], np.repeat(ts, len(ks)))
                or not np.array_equal(quotes[:, 3], np.tile(ks, len(ts)))
            ):
                raise ValueError("Incomplete or reordered implied-volatility CSV surface")
            # Use the saved log-moneyness directly: log(exp(k)) would add an
            # unnecessary rounding step when restoring the exact interpolant.
            provenance = dict(
                meta.get("implied_surface_sources", {}).get(
                    key, meta["volatility_model"].get("source", {})
                )
            )
            provenance["reloaded_from_export"] = {"file_name": path.name}
            return DupireLocalVol(
                ts,
                ks,
                quotes[:, 2].reshape(len(ts), len(ks)),
                spot=meta["quoted_spot"],
                source=provenance,
            )

        surface = read_implied("base")
        normalized = NormalizedDupire(surface)
        model_metadata = dict(meta["volatility_model"])
        model_metadata["source"] = dict(surface.source)
        model_metadata["reloaded_from_export"] = True
        default_label = (
            "DUPIRE ON A SYNTHETIC SURFACE - not the paper's Eurostoxx data"
            if meta["model_tag"] == "dupire_demo"
            else "DUPIRE ON SUPPLIED CSV - provenance recorded in metadata"
        )
        model = {
            "base": normalized,
            "surface": surface,
            "tag": meta["model_tag"],
            "status": meta["model_status"],
            "deformation_space": meta["deformation_space"],
            "metadata": model_metadata,
            "label": meta.get("model_label", default_label),
        }
        if meta["deformation_space"] == "implied":
            for key in ("parallel", "skew"):
                model[key] = NormalizedDupire(read_implied(key))
        else:
            model["parallel"] = deformation(normalized, 0.01)
            model["skew"] = lambda tau, spot: normalized(tau, spot) - 0.01 * np.tanh(
                2 * np.log(np.maximum(spot, 1e-8) / 100)
            )
    return {
        "base": base,
        "risk": risk,
        "config": cfg,
        "model": model,
        "metadata": meta,
        "products": [Product(**entry["parameters"]) for entry in meta["products"]],
    }


class Algorithm2Exports:
    """Record every run and write hedge weights during simulation."""

    columns = [
        "run_id",
        "purpose",
        "seed",
        "start_day",
        "start_spot",
        "period_business_days",
        "factors",
        "npaths",
        "calibrate",
        "trace_status",
        "trace_file",
        "trace_rows",
        "qstar",
        "reused_from",
    ]

    def __init__(self, folder, metadata, cfg, model):
        """Store output settings and initialize the index of Monte Carlo runs."""
        self.folder = Path(folder)
        self.metadata = metadata
        self.cfg = cfg
        self.model = model
        self.runs = []
        self.put_ids = metadata["matrix_A_columns"]

    def _register(self, row):
        """Add a unique run to the index and write the updated CSV."""
        if any(r["run_id"] == row["run_id"] for r in self.runs):
            raise ValueError("Duplicate MC run")
        self.runs.append(row)
        write_csv(
            self.folder / "mc_runs.csv",
            self.columns,
            ([r.get(c, "") for c in self.columns] for r in self.runs),
        )

    def simulate(
        self,
        risk,
        cfg,
        start_day=0,
        spot=100.0,
        period=1,
        npaths=None,
        seed=None,
        factors=2,
        calibrate=False,
        run_id=None,
        purpose="auxiliary",
    ):
        """Run the reference simulator, save the requested trace and register the run."""
        if run_id is None:
            raise ValueError("A run identifier is required for export")
        selected = purpose == "principal" or cfg.trace_runs == "all"
        path = export_csv_path(self.folder, f"weights_{run_id}.csv", cfg)
        row = {
            "run_id": run_id,
            "purpose": purpose,
            "seed": cfg.seed if seed is None else seed,
            "start_day": start_day,
            "start_spot": spot,
            "period_business_days": period,
            "factors": factors,
            "npaths": cfg.paths if npaths is None else npaths,
            "calibrate": int(calibrate),
            "trace_status": "complete" if selected else "not_requested",
            "trace_file": "",
            "trace_rows": 0,
            "qstar": "",
            "reused_from": "",
        }
        kwargs = dict(
            start_day=start_day,
            spot=spot,
            period=period,
            npaths=npaths,
            seed=seed,
            factors=factors,
            calibrate=calibrate,
            sigma=self.model["base"],
        )
        if selected:
            log(f"[Algorithm 2 CSV] MC weights: {run_id}, all {row['npaths']} paths")
            with CsvMcTrace(path, self.put_ids) as trace:
                result = simulate_fhc(risk, cfg, trace=trace, **kwargs)
            row["trace_file"] = path.relative_to(self.folder).as_posix()
            row["trace_rows"] = trace.rows
        else:
            result = simulate_fhc(risk, cfg, **kwargs)
        row["qstar"] = result["qstar"] if result["qstar"] is not None else ""
        self._register(row)
        result["export_run_id"] = run_id
        return result

    def reuse(self, run_id, purpose, result):
        """Register a result already generated by another run."""
        source = next(r for r in self.runs if r["run_id"] == result["export_run_id"])
        row = dict(source)
        row.update(
            run_id=run_id,
            purpose=purpose,
            reused_from=source["run_id"],
            trace_status="reused_from:" + source["run_id"],
        )
        self._register(row)
        return result

    def finish(self, qstar):
        """Save the final calibration and run index, then return the export summary."""
        self.metadata["qstar_decimal_vega_per_unit_notional"] = qstar
        self.metadata["mc_runs"] = self.runs
        write_json(self.folder / "algorithm2_metadata.json", self.metadata)
        return {
            "directory": str(self.folder),
            "metadata": "algorithm2_metadata.json",
            "run_index": "mc_runs.csv",
            "trace_scope": self.cfg.trace_runs,
            "csv_compression": self.cfg.csv_compression,
            "simulation_count": len(self.runs),
        }


class CsvMcTrace:
    """Write every path's initial hedge, actual updates and termination events.

    event_code:
      0 initial hedge; dw=w0, turnover zero because initial costs are excluded;
      1 attempted hedge correction (the risk threshold triggered do=True);
      2 early autocall with unwind; economic w_after=0 and real turnover;
      3 maturity; economic w_after=0, dw=-w_before, no trading turnover;
      4 early autocall without unwind; weights unchanged, tracking terminates.

    hedge_tracked_after=0 means the simulation no longer follows that hedge.
    Under event4 it does not imply that the remaining options were liquidated.
    Event3 describes expiry economically without changing the MC's internal w.
    The context manager closes the output on normal exit and on any exception.
    """

    PREFIX = (
        "path_id",
        "day",
        "time_years",
        "spot",
        "event_code",
        "note_alive_after",
        "hedge_tracked_after",
        "turnover_q",
        "turnover_q2",
        "cumulative_q",
        "cumulative_q2",
        "rehedges_cumulative",
    )

    def __init__(self, path, put_ids):
        """Set the trace destination and initialize row and event counters."""
        self.path = Path(path)
        self.put_ids = tuple(str(x) for x in put_ids)
        if (
            not self.put_ids
            or any(not x for x in self.put_ids)
            or len(set(self.put_ids)) != len(self.put_ids)
        ):
            raise ValueError("put_ids must contain unique, nonempty identifiers.")
        self._stream = None
        self._started = False
        self._rows = 0
        self._event_counts = {i: 0 for i in range(5)}

    @property
    def rows(self):
        """Return the number of emitted path-event rows."""
        return self._rows

    @property
    def event_counts(self):
        """Return the row count by event type."""
        return dict(self._event_counts)

    def __enter__(self):
        """Open the trace file, write column names and return the writer."""
        if self._started:
            raise RuntimeError("A CsvMcTrace instance can be opened only once.")
        self._started = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open_export_csv(self.path, "wt")
        try:
            header = (
                list(self.PREFIX)
                + ["w_" + x for x in self.put_ids]
                + ["dw_" + x for x in self.put_ids]
            )
            csv.writer(self._stream, lineterminator="\n").writerow(header)
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        """Close the trace stream if it is open."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the trace stream when leaving the context."""
        self.close()
        return False

    def emit(
        self,
        path_ids,
        day,
        spots,
        event_code,
        note_alive_after,
        hedge_tracked_after,
        turnover_q,
        turnover_q2,
        cumulative_q,
        cumulative_q2,
        rehedges_cumulative,
        weights,
        delta_weights,
    ):
        """Write one event batch in bounded blocks, with 17 significant digits."""
        if self._stream is None:
            raise RuntimeError("CsvMcTrace must be used inside a with block.")
        ids = np.asarray(path_ids)
        if ids.ndim != 1:
            raise ValueError("path_ids must be a vector.")
        n = ids.size
        if not n:
            return
        if event_code not in self._event_counts:
            raise ValueError("Invalid event_code.")
        w = np.asarray(weights)
        dw = np.asarray(delta_weights)
        shape = (n, len(self.put_ids))
        if w.shape != shape or dw.shape != shape:
            raise ValueError(f"Invalid trace weight shape: expected {shape}.")
        # Broadcast scalar flags/zero costs as views; do not alter caller arrays.
        values = [
            np.broadcast_to(np.asarray(x, dtype=float), (n,))
            for x in [
                spots,
                note_alive_after,
                hedge_tracked_after,
                turnover_q,
                turnover_q2,
                cumulative_q,
                cumulative_q2,
                rehedges_cumulative,
            ]
        ]
        try:
            for first in range(0, n, 4096):
                last = min(first + 4096, n)
                sl = slice(first, last)
                block = np.empty((last - first, len(self.PREFIX) + 2 * len(self.put_ids)))
                block[:, 0] = ids[sl]
                block[:, 1] = day
                block[:, 2] = day / 252
                block[:, 3] = values[0][sl]
                block[:, 4] = event_code
                for column, value in enumerate(values[1:], start=5):
                    block[:, column] = value[sl]
                block[:, 12 : 12 + len(self.put_ids)] = w[sl]
                block[:, 12 + len(self.put_ids) :] = dw[sl]
                np.savetxt(self._stream, block, fmt="%.17g", delimiter=",")
                count = last - first
                self._rows += count
                self._event_counts[event_code] += count
        except BaseException:
            self.close()
            raise


def simulate_fhc(
    risk,
    cfg,
    start_day=0,
    spot=100.0,
    period=1,
    npaths=None,
    seed=None,
    factors=2,
    calibrate=False,
    sigma=None,
    trace=None,
):
    """Algorithm2, with optional lossless logging of the actual hedge decisions.

    Numerical operations and RNG consumption are identical to the original
    simulate_fhc. Trace-only maturity weights are not written into internal w.
    """
    sigma = proxy_sigma if sigma is None else sigma
    n = cfg.paths if npaths is None else npaths
    seed = cfg.seed if seed is None else seed
    if not 0 <= start_day <= 1260:
        raise ValueError("start_day is outside [0,1260]")
    if period < 1:
        raise ValueError("period must be positive")
    t0 = start_day / 252
    horizon = 5 - t0
    s = np.full(n, spot, dtype=float)
    alive = np.ones(n, dtype=bool)
    duration = np.full(n, horizon)
    total = np.zeros(n)
    squared = np.zeros(n)
    rehedges = np.zeros(n, dtype=int)
    w = np.zeros((n, 14))
    if start_day == 1260:
        if trace is not None:
            trace.emit(
                np.arange(n),
                1260,
                s,
                3,
                0,
                0,
                0.0,
                0.0,
                total,
                squared,
                rehedges,
                w,
                np.zeros_like(w),
            )
        return dict(
            q=total,
            q2=squared,
            duration=np.zeros(n),
            rehedges=rehedges,
            qstar=None,
            outside_count=0,
            post_residual_max=0.0,
            threshold_breaches_after_update=0,
        )
    v0 = risk.sample(t0, [spot])[0]
    A0 = v0[:factors, 1:]
    b0 = v0[:factors, 0]
    w[:] = np.linalg.pinv(A0, rcond=cfg.rcond) @ b0
    if trace is not None:
        trace.emit(
            np.arange(n), start_day, s, 0, 1, 1, 0.0, 0.0, total, squared, rehedges, w, w
        )
    prevvega = np.full(n, v0[0, 0] / 0.01)
    rng = np.random.default_rng(seed)
    calsum = 0.0
    calcount = 0
    outside = 0
    maxres = 0.0
    unsatisfied = 0
    dt = 1 / 252
    rootdt = np.sqrt(dt)
    for day in range(start_day + 1, 1261):
        if not np.any(alive):
            break
        # Draw for ALL paths on ALL business days: common seed really gives
        # common Brownian paths across different hedging periods.
        sig = sigma((day - 1) / 252, s)
        s *= np.exp(-0.5 * sig**2 * dt + sig * rootdt * rng.standard_normal(n))
        was_alive = alive.copy()
        annual = day % 252 == 0
        called = alive & (s >= 100) if annual and day < 1260 else np.zeros(n, dtype=bool)
        expired = alive.copy() if day == 1260 else np.zeros(n, dtype=bool)
        ending = called | expired
        duration[ending] = (day - start_day) / 252
        alive[ending] = False
        # Terminations not requiring risk interpolation must be logged before
        # the possible empty-ids continue. No internal holdings are changed.
        if trace is not None:
            expiry_ids = np.flatnonzero(expired)
            if expiry_ids.size:
                trace.emit(
                    expiry_ids,
                    day,
                    s[expiry_ids],
                    3,
                    0,
                    0,
                    0.0,
                    0.0,
                    total[expiry_ids],
                    squared[expiry_ids],
                    rehedges[expiry_ids],
                    np.zeros_like(w[expiry_ids]),
                    -w[expiry_ids],
                )
            if not cfg.include_unwind:
                no_unwind_ids = np.flatnonzero(called)
                if no_unwind_ids.size:
                    trace.emit(
                        no_unwind_ids,
                        day,
                        s[no_unwind_ids],
                        4,
                        0,
                        0,
                        0.0,
                        0.0,
                        total[no_unwind_ids],
                        squared[no_unwind_ids],
                        rehedges[no_unwind_ids],
                        w[no_unwind_ids],
                        np.zeros_like(w[no_unwind_ids]),
                    )
        rebalance = (day - start_day) % period == 0
        need = (
            was_alive
            if calibrate
            else (called if cfg.include_unwind else np.zeros(n, dtype=bool))
            | (alive if rebalance else np.zeros(n, dtype=bool))
        )
        ids = np.flatnonzero(need)
        if not ids.size:
            continue
        rr = risk.sample(day / 252, s[ids])
        out = s[ids] > risk.s[-1]
        outside += int(out.sum())
        rr[out] = 0
        vega = np.abs(rr[:, 0, 1:] / 0.01)
        if cfg.include_unwind:
            c = called[ids]
            if np.any(c):
                if trace is not None:
                    unwind_delta = -w[ids[c]].copy()
                q = np.sum(np.abs(w[ids[c]]) * vega[c], axis=1)
                total[ids[c]] += q
                squared[ids[c]] += q * q
                w[ids[c]] = 0
                if trace is not None:
                    ii = ids[c]
                    trace.emit(
                        ii,
                        day,
                        s[ii],
                        2,
                        0,
                        0,
                        q,
                        q * q,
                        total[ii],
                        squared[ii],
                        rehedges[ii],
                        w[ii],
                        unwind_delta,
                    )
        if rebalance:
            z = alive[ids]
            if np.any(z):
                ii = ids[z]
                mat = rr[z, :factors, 1:]
                target = rr[z, :factors, 0]
                new, dw, do = minimum_change(mat, target, w[ii], cfg.rcond, cfg.loss_tolerance)
                q = np.sum(np.abs(dw) * vega[z], axis=1)
                total[ii] += q
                squared[ii] += q * q
                rehedges[ii] += do
                w[ii] = new
                if np.any(do):
                    residual = target[do] - np.einsum("nij,nj->ni", mat[do], new[do])
                    peak = np.max(np.abs(residual), axis=1)
                    maxres = max(maxres, float(peak.max()))
                    unsatisfied += int(np.sum(peak > cfg.loss_tolerance))
                    if trace is not None:
                        updated = ii[do]
                        trace.emit(
                            updated,
                            day,
                            s[updated],
                            1,
                            1,
                            1,
                            q[do],
                            q[do] * q[do],
                            total[updated],
                            squared[updated],
                            rehedges[updated],
                            new[do],
                            dw[do],
                        )
        if calibrate:
            newvega = rr[:, 0, 0] / 0.01
            newvega[ending[ids]] = 0
            calsum += float(np.sum(np.abs(newvega - prevvega[ids])))
            calcount += len(ids)
            prevvega[ids] = newvega
    return dict(
        q=total,
        q2=squared,
        duration=duration,
        rehedges=rehedges,
        qstar=calsum / calcount if calcount else None,
        outside_count=outside,
        post_residual_max=maxres,
        threshold_breaches_after_update=unsatisfied,
    )


def stats(values):
    """Summarize a sample by its mean, standard error, confidence interval and quantiles."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return dict(
        mean=mean,
        standard_error=se,
        ci95_low=mean - 1.96 * se,
        ci95_high=mean + 1.96 * se,
        n=len(values),
        median=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
    )


def section4(cfg, out, model=None, results_dir=None):
    """Run the Section 4 hedging-cost experiments and save their tables, grids and figures."""
    results_dir = out.parent if results_dir is None else Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = out / "data"
    data.mkdir(exist_ok=True)
    if model is None:
        model = proxy_model()
    tag = model["tag"]
    label = model["label"]
    log(f"[Section 4 {tag}] {label}")
    products = [Product("autocall", 5, coupon=0.03, coupon_mode=cfg.coupon_mode)] + [
        Product("put", 5, k) for k in range(30, 161, 10)
    ]
    base, risk = risk_bundle(products, [model["base"], model["parallel"], model["skew"]], cfg)
    exports = save_results_csv(base, risk, products, model, cfg, results_dir)
    simulate = exports.simulate
    startprice = float(base.sample(0, [100])[0, 0])
    del base
    gc.collect()
    log("[Section 4 FHC] Independent pilot for quadratic-impact calibration")
    pilot = simulate(
        risk,
        cfg,
        npaths=cfg.paths,
        seed=cfg.seed + 901,
        factors=2,
        calibrate=True,
        run_id="pilot_impact",
        purpose="pilot",
    )
    qstar = pilot["qstar"]
    if qstar is None or not np.isfinite(qstar) or qstar <= 0:
        raise RuntimeError("Nonpositive Qstar calibration")
    log(f"[Section 4 FHC] Qstar={qstar:.8g} decimal vega per unit notional")
    two = simulate(risk, cfg, factors=2, run_id="main_parallel_skew", purpose="principal")
    one = simulate(risk, cfg, factors=1, run_id="main_parallel", purpose="principal")
    const = cfg.spread * two["q"] * 1e4
    linear = cfg.spread * two["q2"] / qstar * 1e4
    constant_one = cfg.spread * one["q"] * 1e4
    summary = {
        "status": model["status"],
        "initial_price": startprice,
        "volatility_model": model["metadata"],
        "figure_suffix": tag,
        "output_directory": out.name,
        "figure_label": label,
        "qstar_decimal_vega_per_unit_notional": qstar,
        "qstar_pilot_paths": cfg.paths,
        "qstar_pilot_seed": cfg.seed + 901,
        "constant_parallel_only_bp": stats(constant_one),
        "constant_parallel_skew_bp": stats(const),
        "linear_impact_parallel_skew_bp": stats(linear),
        "duration_years": stats(two["duration"]),
        "published_reference_not_calibration_targets": {
            "parallel_only_bp": 86,
            "parallel_skew_bp": 91,
            "linear_impact_tail_above_342bp_percent": 1.5,
        },
        "proxy_tail_above_342bp_percent": float(100 * np.mean(linear > 342)),
        "out_of_domain_path_time_count": int(
            one["outside_count"] + two["outside_count"] + pilot["outside_count"]
        ),
        "max_post_update_scenario_residual": max(
            one["post_residual_max"], two["post_residual_max"]
        ),
        "threshold_breaches_after_update": one["threshold_breaches_after_update"]
        + two["threshold_breaches_after_update"],
    }
    write_csv(
        data / "fhc_pathwise.csv",
        [
            "path",
            "constant_parallel_bp",
            "constant_parallel_skew_bp",
            "linear_impact_bp",
            "duration_years",
            "rehedges",
        ],
        zip(range(cfg.paths), constant_one, const, linear, two["duration"], two["rehedges"]),
    )

    # Profile times include both sides of every observation, never the ambiguous
    # conditioning S=barrier exactly at the observation itself.
    days = {0, 1260}
    for y in range(5):
        for q in range(1, cfg.profile_times_year):
            days.add(int(round((y + q / cfg.profile_times_year) * 252)))
    for y in range(1, 5):
        days.update([y * 252 - 1, y * 252 + 1])
    rows = []
    log(f"[Section 4 FHC] ATM profiles: {len(days)} dates, {cfg.profile_paths} paths per date")
    for idx, day in enumerate(sorted(days)):
        run_id = f"profile_atm_day_{day:04d}"
        if day == 0 and cfg.profile_paths == cfg.paths:
            r = exports.reuse(run_id, "profile_atm", one)
        else:
            r = simulate(
                risk,
                cfg,
                day,
                npaths=cfg.profile_paths,
                seed=cfg.seed + 123,
                factors=1,
                run_id=run_id,
                purpose="profile_atm",
            )
        cc = stats(cfg.spread * r["q"] * 1e4)
        dd = stats(r["duration"])
        rows.append(
            [day / 252, cc["mean"], cc["standard_error"], dd["mean"], dd["standard_error"]]
        )
        summary["out_of_domain_path_time_count"] += r["outside_count"]
        if idx % 5 == 0:
            log(f"  ATM profile {idx+1}/{len(days)}")
    rows = np.asarray(rows)
    write_csv(
        data / "fig10_11_profiles.csv",
        ["time_years", "fhc_bp", "fhc_se_bp", "duration_years", "duration_se"],
        rows,
    )
    for number, col, title, ylabel in [
        (10, 1, "ATM FHC conditional on survival", "FHC (bp)"),
        (11, 3, "Expected remaining lifetime at the money", "Remaining years"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.7))
        ax.plot(rows[:, 0], rows[:, col], "o-", ms=3, lw=1.3, label="MC: mean")
        ax.fill_between(
            rows[:, 0],
            rows[:, col] - 1.96 * rows[:, col + 1],
            rows[:, col] + 1.96 * rows[:, col + 1],
            alpha=0.18,
            label="Indicative 95% MC interval",
        )
        for yr in range(1, 5):
            ax.axvline(yr, color="grey", alpha=0.2, lw=0.7)
        ax.set(xlabel="t (years)", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig_label(fig, number, title, label)
        fig.subplots_adjust(top=0.85, bottom=0.15)
        save_reproduced_figure(
            fig, "autocall_5_years", f"fig_{number:02d}_{tag}.png", cfg, results_dir
        )

    fig, ax = plt.subplots(figsize=(9, 5.7))
    spotrows = []
    spots = np.linspace(10, 125, cfg.profile_spots)
    for day, date_label in [
        (1, "1 day after issuance"),
        (251, "1 day before the first autocall observation"),
        (253, "1 day after the first autocall observation"),
    ]:
        log(f"[Section 4 FHC] Spot profile: {date_label}")
        means = []
        ses = []
        for spot_index, spot in enumerate(spots):
            r = simulate(
                risk,
                cfg,
                day,
                spot,
                npaths=cfg.profile_paths,
                seed=cfg.seed + 200,
                factors=1,
                run_id=f"profile_spot_day_{day:04d}_index_{spot_index:03d}",
                purpose="profile_spot",
            )
            st = stats(cfg.spread * r["q"] * 1e4)
            means.append(st["mean"])
            ses.append(st["standard_error"])
            spotrows.append([day / 252, spot / 100, st["mean"], st["standard_error"]])
            summary["out_of_domain_path_time_count"] += r["outside_count"]
        ax.plot(spots / 100, means, "o-", ms=3, label=date_label)
        ax.fill_between(
            spots / 100,
            np.array(means) - 1.96 * np.array(ses),
            np.array(means) + 1.96 * np.array(ses),
            alpha=0.08,
        )
    ax.set(xlabel="Spot ratio S/S0", ylabel="FHC (bp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig_label(fig, 12, "FHC against spot on three dates", label)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    save_reproduced_figure(fig, "autocall_5_years", f"fig_12_{tag}.png", cfg, results_dir)
    write_csv(
        data / "fig12_spot_profiles.csv",
        ["time_years", "spot_ratio", "fhc_bp", "fhc_se_bp"],
        spotrows,
    )

    # Figure13 requires implied volatility, NOT plotting raw local volatility.
    log("[Section 4 FHC] Implied smile: PDE call prices and Black-Scholes inversion")
    strikes = np.arange(30, 161, 5)
    ivs = []
    for sig in [model["base"], model["skew"], model["parallel"]]:
        g = solve_pde([Product("call", 1, float(k)) for k in strikes], sig, cfg)
        prices = g.sample(0, [100])[0]
        vol = []
        for k, p in zip(strikes, prices):
            intrinsic = max(100 - k, 0)
            if not intrinsic < p < 100:
                vol.append(np.nan)
            else:
                vol.append(brentq(lambda v: float(bs_price(100, k, 1, v)) - p, 1e-5, 3.0))
        ivs.append(vol)
        del g
    fig, ax = plt.subplots(figsize=(9, 5.7))
    for v, legend in zip(
        ivs,
        [
            "Base " + tag,
            "Skew deformation",
            f"Parallel {model['deformation_space']} bump +0.01",
        ],
    ):
        ax.plot(strikes, 100 * np.array(v), label=legend)
    ax.set(xlabel="Strike (% of S0)", ylabel="1Y implied volatility (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig_label(fig, 13, "Implied smile recovered from the PDE", label)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    save_reproduced_figure(fig, "autocall_5_years", f"fig_13_{tag}.png", cfg, results_dir)
    write_csv(
        data / "fig13_implied_vol.csv",
        ["strike_percent", "base_iv_decimal", "skew_iv_decimal", "parallel_iv_decimal"],
        zip(strikes, *ivs),
    )

    # EUR axis uses illustrative notional100m; cost curve is the chosen convention.
    notional = 1e8
    qstar_eur = qstar * 0.01 * notional
    q_eur = np.linspace(0, max(20000, 2.5 * qstar_eur), 301)
    q_decimal = q_eur / 0.01
    cc = cfg.spread * q_decimal
    ll = cfg.spread * q_decimal * q_decimal / (qstar * notional)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    ax.plot(q_eur, cc, label="Constant spread: linear cost")
    ax.plot(q_eur, ll, label="Linear impact: quadratic cost")
    ax.axvline(
        qstar_eur, ls=":", color="grey", label=f"Q* calibration: {qstar_eur:,.0f} EUR/vol-point"
    )
    ax.set(xlabel="Traded vega (EUR per 1 volatility percentage point)", ylabel="Cost (EUR)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig_label(fig, 14, "Cost functions, illustrative EUR 100m notional", label)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    save_reproduced_figure(fig, "autocall_5_years", f"fig_14_{tag}.png", cfg, results_dir)
    write_csv(
        data / "fig14_cost_functions.csv",
        ["vega_EUR_per_volpoint", "constant_cost_EUR", "linear_impact_cost_EUR"],
        zip(q_eur, cc, ll),
    )

    # Explicit overflow, as in the paper; full data remain available in CSV.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"width_ratios": [3, 1]})
    bins = np.linspace(0, 342, 58)
    axes[0].hist(
        const[const < 342], bins=bins, histtype="step", lw=1.7, label="Constant spread"
    )
    axes[0].hist(
        linear[linear < 342], bins=bins, histtype="step", lw=1.7, label="Linear impact"
    )
    axes[0].set(xlabel="FHC (bp), values <342", ylabel="Number of paths")
    axes[0].legend(fontsize=8)
    counts = [int(np.sum(const >= 342)), int(np.sum(linear >= 342))]
    axes[1].bar([0, 1], counts, color=["#1f77b4", "#ff7f0e"])
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Constant", "Linear"],
        ylabel="Number of paths",
        title="Tail: FHC >=342 bp",
    )
    for x, y in enumerate(counts):
        axes[1].text(x, y, f" {100*y/cfg.paths:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(1, max(counts)) * 1.2)
    fig_label(fig, 15, "FHC distribution with explicit tail counts", label)
    fig.subplots_adjust(top=0.82, bottom=0.17, wspace=0.32)
    save_reproduced_figure(fig, "autocall_5_years", f"fig_15_{tag}.png", cfg, results_dir)

    periods = [1, 2, 5, 10, 15, 20, 25, 30]
    periodrows = []
    for p in periods:
        log(f"[Section 4 FHC] Rehedging period: {p} days")
        run_id = f"frequency_{p:03d}_days"
        result = (
            exports.reuse(run_id, "frequency", two)
            if p == 1
            else simulate(risk, cfg, period=p, factors=2, run_id=run_id, purpose="frequency")
        )
        a = stats(cfg.spread * result["q"] * 1e4)
        b = stats(cfg.spread * result["q2"] / qstar * 1e4)
        periodrows.append([p, a["mean"], a["standard_error"], b["mean"], b["standard_error"]])
        summary["out_of_domain_path_time_count"] += result["outside_count"]
    periodrows = np.array(periodrows)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    for col, cost_label in [(1, "Constant spread"), (3, "Linear impact; fixed daily Q*")]:
        ax.errorbar(
            periodrows[:, 0],
            periodrows[:, col],
            yerr=1.96 * periodrows[:, col + 1],
            fmt="o-",
            capsize=3,
            label=cost_label,
        )
    ax.set(xlabel="Rehedging period (business days)", ylabel="FHC (bp)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig_label(fig, 16, "Effect of hedging frequency", label)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    save_reproduced_figure(fig, "autocall_5_years", f"fig_16_{tag}.png", cfg, results_dir)
    write_csv(
        data / "fig16_frequency.csv",
        [
            "period_business_days",
            "constant_mean_bp",
            "constant_se_bp",
            "linear_impact_mean_bp",
            "linear_impact_se_bp",
        ],
        periodrows,
    )
    if isinstance(model["base"], NormalizedDupire):
        summary["dupire_boundary_evaluations_total"] = dict(model["base"].counts)
    summary["algorithm2_exports"] = exports.finish(qstar)
    write_json(out / "summary.json", summary)
    return summary


def reference_upin(s=100.0, k=100.0, barrier=120.0, tau=2.0, sigma=0.3):
    """Independent reflection-principle quadrature, used ONLY to verify PDE.

    For x=log(S), b=log(B), v=sigma^2*tau, the killed log density below b is
    normal(y;x-v/2,v)*[1-exp(2*(b-x)*(y-b)/v)]. Integrating the vanilla
    payoff against it gives up-and-out; vanilla minus up-and-out gives in.
    This is our independent derivation, not an extra equation in the paper.
    """
    from scipy.integrate import quad

    if s >= barrier:
        return float(bs_price(s, k, tau, sigma))
    x = np.log(s)
    b = np.log(barrier)
    v = sigma * sigma * tau

    def f(y):
        """Evaluate the up-and-out pricing integrand for the independent parity check."""
        density = np.exp(-0.5 * (y - x + v / 2) ** 2 / v) / np.sqrt(2 * np.pi * v)
        return (np.exp(y) - k) * density * (-np.expm1(2 * (b - x) * (y - b) / v))

    out = quad(f, np.log(k), b, epsabs=1e-11, epsrel=1e-11)[0]
    return float(bs_price(s, k, tau, sigma)) - out


def reference_autocall(s=100.0, sigma=0.30, t=0.0):
    """Independent Gaussian quadrature: 3-year note, coupons10/20/30%.

    Exact final transition via CDF; preceding transitions integrated only
    on the survival domain, with 80 Gauss-Legendre nodes and 12-sigma tails.
    Used solely for validation, not as a substitute for paper's PDE method.
    """
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(80)
    events = [(year, 0.1 * year) for year in [1, 2, 3] if year > t]

    def value(spot, now, idx):
        """Value the remaining autocall cash flows recursively at an observation step."""
        date, coupon = events[idx]
        v = sigma * sigma * (date - now)
        std = np.sqrt(v)
        d2 = (np.log(np.maximum(spot, 1e-300) / 100) - v / 2) / std
        if idx == len(events) - 1:
            db = (np.log(np.maximum(spot, 1e-300) / 70) - v / 2) / std
            return 1 + coupon * ndtr(d2) - ndtr(-db) + (spot / 100) * ndtr(-db - std)
        upper = np.clip(-d2, -12, 12)
        half = (upper + 12) / 2
        mid = (upper - 12) / 2
        z = mid[..., None] + half[..., None] * nodes
        nxt = spot[..., None] * np.exp(-v / 2 + std * z)
        cont = value(nxt, date, idx + 1)
        return (1 + coupon) * ndtr(d2) + half * np.sum(
            cont * np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * weights, axis=-1
        )

    return float(value(np.asarray(s, dtype=float), t, 0))


def run_validation(out):
    """Independent pricing, convergence, events, linear algebra, units."""
    log("[Validation] Independent prices, convergence, events, hedges and units")
    result = {"checks": [], "pricing": []}

    def check(name, ok, detail):
        """Record the check name, pass flag and diagnostic details."""
        result["checks"].append({"name": name, "passed": bool(ok), "detail": detail})
        if not ok:
            log("VALIDATION FAILED: " + name)

    refs = [
        float(bs_price(100, 100, 2, 0.3)),
        float(bs_price(100, 100, 2, 0.3, "digital")),
        reference_upin(),
        reference_autocall(),
    ]
    errors = []
    for ds, steps in [(1.0, 126), (0.5, 252)]:
        c = Config(ds=ds, steps_year=steps)
        g = solve_pde([Product("call", 2), Product("digital", 2)], flat_sigma(0.3), c)
        plain = g.sample(0, [100])[0]
        b = barrier_solver(flat_sigma(0.3), c).sample(0, [100])[0, 0]
        a = solve_pde([Product("autocall", 3), Product("call", 2)], flat_sigma(0.3), c)
        auto = a.sample(0, [100])[0, 0]
        found = [float(plain[0]), float(plain[1]), float(b), float(auto)]
        errs = np.abs(np.array(found) - refs)
        errors.append(errs)
        result["pricing"].append(
            {
                "ds": ds,
                "steps_per_year": steps,
                "products": ["call", "digital", "upin", "autocall"],
                "pde": found,
                "independent_reference": refs,
                "absolute_error": errs,
            }
        )
        post = a.sample(2, [100, 575, 590, 600], side="post")[:, 1]
        check(
            f"expired_call_post_event_ds_{ds}",
            np.max(np.abs(post)) < 1e-12,
            {"max": float(np.max(np.abs(post)))},
        )
        pre = a.sample(1, [100, 120], side="pre")[:, 0]
        check(
            f"autocall_redemption_ds_{ds}", np.max(np.abs(pre - 1.1)) < 1e-12, {"values": pre}
        )
    check(
        "mesh_convergence_all_products",
        np.all(errors[1] < 0.65 * errors[0]),
        {"coarse_errors": errors[0], "fine_errors": errors[1]},
    )
    check(
        "pricing_absolute_tolerances",
        np.all(errors[1] < np.array([0.001, 0.00003, 0.001, 0.002])),
        {"fine_errors": errors[1]},
    )

    rng = np.random.default_rng(2023)
    A = rng.normal(size=(20, 2, 14))
    b = rng.normal(size=(20, 2))
    old = rng.normal(size=(20, 14))
    new, delta, do = minimum_change(A, b, old, tolerance=0)
    res = np.max(np.abs(np.einsum("nij,nj->ni", A, new) - b))
    check("minimum_change_constraint", res < 1e-11, {"max_residual": float(res)})
    proj = np.linalg.pinv(A) @ A
    null_old = old - np.einsum("nij,nj->ni", proj, old)
    null_new = new - np.einsum("nij,nj->ni", proj, new)
    err = float(np.max(np.abs(null_new - null_old)))
    check("minimum_change_nullspace_preserved", err < 1e-11, {"max_error": err})
    same, _, do = minimum_change(A, np.einsum("nij,nj->ni", A, old), old)
    check("no_turnover_below_threshold", np.array_equal(same, old) and not np.any(do), {})
    # Nonsymmetric random matrix catches accidental transpose in SVD formula.
    aa = rng.normal(size=(4, 5, 2, 2))
    known = rng.normal(size=(4, 5, 2))
    bb = np.einsum("...ij,...j->...i", aa, known)
    grid = Grid(np.arange(4), np.arange(5), np.concatenate([bb[..., None], aa], axis=-1), {})
    estimated, _, _ = solve_weights(grid, Config())
    err = float(np.nanmax(np.abs(estimated - known)))
    check("svd_weight_orientation", err < 1e-10, {"max_error": err})
    cost = 0.002 * (20000 / 0.01)
    check("cash_vega_units_20000_eur_to_4000_eur", abs(cost - 4000) < 1e-12, {"cost_eur": cost})
    qstar = 0.02
    check(
        "cost_calibration_crossover",
        np.isclose(0.002 * qstar, 0.002 * qstar * qstar / qstar),
        {},
    )
    # In the infinitesimal limit both temporal directions are multiples of BS vega.
    vega = np.array(
        [
            np.exp(-0.5 * ((np.log(100 / k) + 0.3**2) / (0.3 * np.sqrt(2))) ** 2)
            * 100
            * np.sqrt(2 / (2 * np.pi))
            for k in [100, 110]
        ]
    )
    limiting = np.outer([1, (1 - np.exp(-2)) / 2], vega)
    check(
        "same_expiry_gateaux_rank_one",
        np.linalg.matrix_rank(limiting) == 1,
        {"singular_values": np.linalg.svd(limiting, compute_uv=False)},
    )
    # Source discrepancy: use independent variance-clock quadrature, not PDE,
    # to verify the negative residual implied by the printed setup of Fig5(c).
    residual_refs = []
    for now in [0.0, 1.0]:
        tau = 2 - now
        sig = 0.2
        eps = 0.002
        u = reference_upin(tau=tau, sigma=sig)
        h = float(bs_price(100, 120, tau, sig))
        w = (reference_upin(tau=tau, sigma=sig + eps) - u) / (
            float(bs_price(100, 120, tau, sig + eps)) - h
        )
        variance = (
            sig * sig * tau
            + 2 * sig * eps * (np.exp(-now) - np.exp(-2))
            + eps * eps * (np.exp(-2 * now) - np.exp(-4)) / 2
        )
        effective = np.sqrt(variance / tau)
        du = reference_upin(tau=tau, sigma=effective) - u
        dh = float(bs_price(100, 120, tau, effective)) - h
        rr = 100 * (w * dh - du) / abs(du)
        residual_refs.append(
            {"time": now, "spot": 100, "residual_percent": float(rr), "hedge_weight": float(w)}
        )
    result["figure5c_independent_reference_and_source_discrepancy"] = residual_refs
    check(
        "figure5c_printed_setup_negative_residual",
        all(x["residual_percent"] < 0 for x in residual_refs),
        {
            "note": "This verifies the printed setup, not agreement with the published positive-looking surface."
        },
    )
    result["passed"] = all(x["passed"] for x in result["checks"])
    write_json(Path(out) / "validation.json", result)
    if not result["passed"]:
        raise RuntimeError("Validation failed: see validation.json")
    return result


# ORIGINAL FIGURE EXTRACTION: source renderings, not recalculated data.
SOURCE_PDF_SHA256 = "29f6d60582e0bd4e1167a20596ab88eecedad6f7864634c8b021896d1a189817"

# Figure number: (zero-based PDF page, crop rectangle in PDF points).
# All rectangles include the complete published caption.
FIGURE_CROPS = {
    1: (5, (116, 310, 480, 487)),
    2: (5, (116, 569, 480, 750)),
    3: (6, (116, 260, 480, 593)),
    4: (7, (116, 569, 480, 750)),
    5: (8, (116, 120, 480, 454)),
    6: (9, (116, 112, 480, 366)),
    7: (9, (116, 454, 480, 750)),
    8: (10, (116, 371, 480, 548)),
    9: (11, (116, 120, 480, 454)),
    10: (14, (116, 464, 480, 750)),
    11: (15, (116, 127, 480, 415)),
    12: (15, (116, 442, 480, 703)),
    13: (17, (116, 123, 480, 352)),
    14: (17, (116, 464, 480, 750)),
    15: (18, (116, 127, 480, 426)),
    16: (18, (116, 464, 480, 750)),
}


def extract_original_figures(pdf_path, out_dir, dpi=160):
    """Render the 16 original figures without redrawing or modifying their data.

    Requires PyMuPDF (``python -m pip install PyMuPDF``). The separate manifest
    labels every image as an original source rendering, records its source page,
    crop rectangle, pixel dimensions, and SHA-256 hashes. Returns that manifest.
    Different source bytes produce a warning. A changed page count, title, or
    absent expected caption raises an error; no guessed crops are emitted.
    """
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "Extracting the original figures requires PyMuPDF: " "python -m pip install PyMuPDF"
        ) from exc

    pdf_path = Path(pdf_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be between 72 and 600")
    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if source_sha256 != SOURCE_PDF_SHA256:
        warnings.warn(
            "The PDF bytes differ from the attachment used to define the crops. "
            "Visually inspect the extracted original figures.",
            RuntimeWarning,
        )

    manifest = {
        "source_filename": pdf_path.name,
        "source_sha256": source_sha256,
        "same_bytes_as_provided_attachment": source_sha256 == SOURCE_PDF_SHA256,
        "kind": "original_figures_rendered_from_user_supplied_pdf",
        "not_numerical_reproductions": True,
        "dpi": dpi,
        "figures": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        if len(doc) != 20:
            raise ValueError(
                "The PDF must be the 20-page version dated 24 August 2023. "
                "The crops are not valid for other page layouts."
            )
        first_page = doc[0].get_text().upper()
        if "THE COST OF VEGA-HEDGING STRUCTURED PRODUCTS" not in first_page:
            raise ValueError("The PDF does not match the expected paper title.")
        for fig_no, (page_index, coords) in FIGURE_CROPS.items():
            page = doc[page_index]
            clip = fitz.Rect(coords)
            caption = f"Figure {fig_no}."
            if caption not in page.get_text(clip=clip):
                raise ValueError(
                    f"Caption {caption} not found in the crop on "
                    f"page {page_index + 1}; the PDF layout differs."
                )
            pix = page.get_pixmap(
                matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
                clip=clip,
                alpha=False,
            )
            image_path = out_dir / f"fig_{fig_no:02d}_original.png"
            pix.save(image_path)
            manifest["figures"].append(
                {
                    "figure": fig_no,
                    "pdf_page": page_index + 1,
                    "crop_pdf_points": list(coords),
                    "filename": image_path.name,
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "kind": "original_source_rendering_not_simulation",
                }
            )
    (out_dir / "original_figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# DUPIRE SURFACE IMPLEMENTATION (local volatility, r=q=0).
"""Dupire LOCAL volatility from a rectangular implied-volatility CSV.

This is local volatility, not a stochastic-local-volatility model. The function
sigma_D(t,S) is deterministic; its value along a random spot path is random.
The existing research engine assumes r=q=0 and spot normalized to100.

Input: maturity (years), strike (same unit as spot), implied_vol (decimal).
Interpolation: C2 tensor cubic spline of total implied variance, s=0. This is
an exact interpolant, NOT a guaranteed arbitrage-free statistical smoother.
Calendar and butterfly conditions are tested densely; invalid surfaces fail.
Finite-grid validation is not a proof of global absence of arbitrage.

Formula: w_T / [(1-k*w_k/(2*w))**2
                  -0.25*(0.25+1/w)*w_k**2+0.5*w_kk], k=ln(K/spot), r=q=0.
All derivatives are derivatives of the spline, not differences of raw quotes.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from scipy.interpolate import RectBivariateSpline, BSpline
from collections import OrderedDict


class DupireSurfaceError(ValueError):
    """The implied surface cannot safely define the configured local model."""


class DupireLocalVol:
    """Callable Dupire local vol with explicit domain and boundary treatment.

    __call__(t,S) accepts broadcastable arrays. S is in the quoted spot units;
    a normalized engine must multiply S by surface.spot/100 before calling.
    For 0<=t<Tmin: w(t,k)=(t/Tmin)*w(Tmin,k); the t=0 local variance is the
    exact limit of that extension. For spot outside quote support, LOCAL vol
    is frozen at the closest k boundary, not extrapolated through a cubic.
    For t>Tmax an exception is raised. No negative variance or denominator is
    clipped into validity. Every bumped surface is rebuilt and revalidated.
    """

    def __init__(
        self,
        maturities,
        log_moneyness,
        implied_vols,
        *,
        source=None,
        spot=100.0,
        denominator_floor=1e-8,
        validate=True,
    ):
        """Validate the quote grid and build the interpolated total-variance surface."""
        self.maturities = np.asarray(maturities, dtype=float)
        self.log_moneyness = np.asarray(log_moneyness, dtype=float)
        self.implied_vols = np.asarray(implied_vols, dtype=float)
        self.spot = float(spot)
        if not np.isfinite(self.spot) or self.spot <= 0:
            raise DupireSurfaceError("Invalid reference spot.")
        self.times = self.maturities
        self.strikes = self.spot * np.exp(self.log_moneyness)
        self.denominator_floor = float(denominator_floor)
        if self.maturities.ndim != 1 or self.log_moneyness.ndim != 1:
            raise DupireSurfaceError("Maturity and log-moneyness must be vectors.")
        if len(self.maturities) < 4 or len(self.log_moneyness) < 5:
            raise DupireSurfaceError(
                "Dupire requires at least 4 maturities and 5 distinct strikes."
            )
        if self.implied_vols.shape != (len(self.maturities), len(self.log_moneyness)):
            raise DupireSurfaceError(
                "The surface must be rectangular: every maturity must include all strikes."
            )
        if not (
            np.all(np.isfinite(self.maturities))
            and np.all(np.isfinite(self.log_moneyness))
            and np.all(np.isfinite(self.implied_vols))
        ):
            raise DupireSurfaceError("The surface contains nonfinite values.")
        if np.any(self.maturities <= 0) or np.any(np.diff(self.maturities) <= 0):
            raise DupireSurfaceError("Maturities must be positive and strictly increasing.")
        if np.any(np.diff(self.log_moneyness) <= 0) or np.any(self.implied_vols <= 0):
            raise DupireSurfaceError(
                "Duplicate or unordered strikes, or nonpositive volatilities."
            )
        if not np.isfinite(self.denominator_floor) or self.denominator_floor <= 0:
            raise DupireSurfaceError("denominator_floor must be positive and finite.")
        self.tmin = float(self.maturities[0])
        self.tmax = float(self.maturities[-1])
        self.kmin = float(self.log_moneyness[0])
        self.kmax = float(self.log_moneyness[-1])
        self.total_variances = self.maturities[:, None] * self.implied_vols**2
        if np.any(np.diff(self.total_variances, axis=0) < -1e-12):
            raise DupireSurfaceError(
                "Calendar arbitrage at the nodes: total variance decreases at fixed k."
            )
        self.spline = RectBivariateSpline(
            self.maturities, self.log_moneyness, self.total_variances, kx=3, ky=3, s=0
        )
        self.source = dict(source or {"kind": "supplied_rectangular_surface"})
        self.validation = {}
        if validate:
            self.validation = self.validate_dense()

    @classmethod
    def from_csv(cls, path, spot=100.0, **kwargs):
        """Read long-format CSV, preserving source bytes and quote provenance.

        spot is the reference level underlying the quoted strikes. Prices in
        the research engine remain normalized to an initial spot of100.
        Current implementation has r=q=0; input quotes must use that convention.
        """
        path = Path(path)
        if not np.isfinite(spot) or spot <= 0:
            raise DupireSurfaceError("The reference spot must be positive.")
        raw = path.read_bytes()
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"maturity", "strike", "implied_vol"}
            if not required.issubset(reader.fieldnames or []):
                raise DupireSurfaceError(
                    "Required CSV columns: maturity,strike,implied_vol; maturity in years and volatility in decimals."
                )
            rows = []
            seen = set()
            for number, row in enumerate(reader, start=2):
                try:
                    values = tuple(
                        float(row[key]) for key in ["maturity", "strike", "implied_vol"]
                    )
                except (TypeError, ValueError) as exc:
                    raise DupireSurfaceError(
                        f"CSV row {number}: invalid numeric values."
                    ) from exc
                t, k, v = values
                if not all(np.isfinite(values)) or t <= 0 or k <= 0 or v <= 0:
                    raise DupireSurfaceError(
                        f"CSV row {number}: maturity, strike and volatility must be positive and finite."
                    )
                key = (t, k)
                if key in seen:
                    raise DupireSurfaceError(f"Duplicate quote at row {number}: T={t}, K={k}.")
                seen.add(key)
                rows.append(values)
        if not rows:
            raise DupireSurfaceError("Empty CSV.")
        ts = np.array(sorted({r[0] for r in rows}))
        ks = np.array(sorted({r[1] for r in rows}))
        if len(rows) != len(ts) * len(ks):
            raise DupireSurfaceError(
                "Incomplete CSV grid: every maturity must contain the same strikes."
            )
        ti = {x: i for i, x in enumerate(ts)}
        ki = {x: i for i, x in enumerate(ks)}
        vols = np.empty((len(ts), len(ks)))
        for t, k, v in rows:
            vols[ti[t], ki[k]] = v
        return cls(
            ts,
            np.log(ks / spot),
            vols,
            source={
                "kind": "csv_implied_volatility_surface",
                "file_name": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "quote_rows": len(rows),
                "quoted_spot": float(spot),
                "engine_spot": 100.0,
                "rates_assumption": "r=q=0",
            },
            spot=spot,
            **kwargs,
        )

    def _coordinates(self, t, k, allow_k_clamp):
        """Validate time and log-moneyness and prepare interpolation coordinates."""
        ta, ka = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(k, dtype=float))
        if np.any(~np.isfinite(ta)) or np.any(~np.isfinite(ka)) or np.any(ta < 0):
            raise DupireSurfaceError("Invalid time or log-moneyness coordinates.")
        if np.any(ta > self.tmax + 1e-10):
            raise DupireSurfaceError(
                f"Maturity exceeds CSV support ({self.tmax:g} years); extrapolation beyond the last maturity is disabled."
            )
        if not allow_k_clamp and np.any((ka < self.kmin - 1e-10) | (ka > self.kmax + 1e-10)):
            raise DupireSurfaceError("Strike lies outside the implied surface support.")
        return np.minimum(ta, self.tmax), np.clip(ka, self.kmin, self.kmax)

    def _fixed_time_splines(self, t):
        """Contract the tensor spline in time once, retaining exact strike splines."""
        if not hasattr(self, "_fixed_time_cache"):
            tx, ty, c = self.spline.tck
            kx, ky = self.spline.degrees
            coefficients = c.reshape(len(tx) - kx - 1, len(ty) - ky - 1)
            self._time_coefficient_spline = BSpline(
                tx, coefficients, kx, axis=0, extrapolate=False
            )
            self._strike_spline_knots = ty
            self._strike_spline_degree = ky
            self._fixed_time_cache = OrderedDict()
        key = float(t)
        cached = self._fixed_time_cache.get(key)
        if cached is not None:
            self._fixed_time_cache.move_to_end(key)
            return cached
        coefficients = self._time_coefficient_spline(key)
        coefficients_dt = self._time_coefficient_spline(key, nu=1)
        strike = BSpline(
            self._strike_spline_knots,
            coefficients,
            self._strike_spline_degree,
            extrapolate=False,
        )
        strike_dt = BSpline(
            self._strike_spline_knots,
            coefficients_dt,
            self._strike_spline_degree,
            extrapolate=False,
        )
        # Derivative splines lower the degree and avoid cancellation of repeatedly
        # differentiating cubic basis values at each spot evaluation.
        cached = (strike, strike_dt, strike.derivative(1), strike.derivative(2))
        self._fixed_time_cache[key] = cached
        if len(self._fixed_time_cache) > 2048:
            self._fixed_time_cache.popitem(last=False)
        return cached

    def components(self, t, k, *, allow_k_clamp=False):
        """Same w,w_T,w_k,w_kk; optimize scalar/fixed times without approximating."""
        ta, ka = self._coordinates(t, k, allow_k_clamp)
        shape = ta.shape
        tf = ta.ravel()
        kf = ka.ravel()
        if tf.size and np.all(tf == tf[0]):
            time = float(tf[0])
            effective = max(time, self.tmin)
            by, by_dt, by_k, by_kk = self._fixed_time_splines(effective)
            w = by(kf)
            wt = by_dt(kf)
            wk = by_k(kf)
            wkk = by_kk(kf)
            if time < self.tmin:
                scale = time / self.tmin
                wt = w / self.tmin
                w = w * scale
                wk = wk * scale
                wkk = wkk * scale
        else:
            te = np.maximum(tf, self.tmin)
            w = self.spline.ev(te, kf)
            wt = self.spline.ev(te, kf, dx=1)
            wk = self.spline.ev(te, kf, dy=1)
            wkk = self.spline.ev(te, kf, dy=2)
            short = tf < self.tmin
            if np.any(short):
                scale = tf[short] / self.tmin
                wt[short] = w[short] / self.tmin
                w[short] *= scale
                wk[short] *= scale
                wkk[short] *= scale
        return tuple(v.reshape(shape) for v in [w, wt, wk, wkk])

    def local_variance(self, t, k, *, allow_k_clamp=True, check=True):
        """Dupire formula; no clipping of invalid local variances."""
        ta, ka = self._coordinates(t, k, allow_k_clamp)
        w, wt, wk, wkk = self.components(ta, ka, allow_k_clamp=True)
        positive = ta > 0
        safe = np.where(positive, w, 1.0)
        g = (1 - ka * wk / (2 * safe)) ** 2 - 0.25 * (0.25 + 1 / safe) * wk**2 + 0.5 * wkk
        if np.any(~positive):
            # Exact limit for w(t,k)=t*a(k). No arbitrary epsilon time is used.
            kk = ka[~positive] if ka.ndim else np.array([float(ka)])
            a = self.spline.ev(np.full(kk.size, self.tmin), kk.ravel()) / self.tmin
            ak = self.spline.ev(np.full(kk.size, self.tmin), kk.ravel(), dy=1) / self.tmin
            limiting = (1 - kk.ravel() * ak / (2 * a)) ** 2
            if g.ndim:
                g[~positive] = limiting
            else:
                g = np.asarray(limiting[0])
        valid = (
            np.isfinite(w)
            & np.isfinite(wt)
            & np.isfinite(g)
            & ((ta == 0) | (w > 0))
            & (wt > 0)
            & (g > self.denominator_floor)
        )
        if check and not np.all(valid):
            raise DupireSurfaceError(
                "Surface is invalid for Dupire: w_T<=0, a nonpositive or too small denominator, or nonfinite values."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            var = wt / g
        if not check:
            var = np.where(valid, var, np.nan)
        return var, g, wt

    def __call__(self, t, s):
        """Evaluate Dupire local volatility at the supplied time and spot."""
        s = np.asarray(s, dtype=float)
        if np.any(~np.isfinite(s)) or np.any(s < 0):
            raise DupireSurfaceError("Invalid spot for local volatility.")
        # S=0 uses the declared left local-vol boundary; diffusion S*sigma=0.
        k = np.log(np.maximum(s, np.finfo(float).tiny) / self.spot)
        variance, _, _ = self.local_variance(t, k, allow_k_clamp=True)
        return np.sqrt(variance)

    def implied_vol(self, t, strike, spot=None):
        """Interpolated input IV inside strike support; T>0 is required."""
        spot = self.spot if spot is None else spot
        ta, ka = np.broadcast_arrays(
            np.asarray(t, dtype=float), np.log(np.asarray(strike, dtype=float) / spot)
        )
        if np.any(ta <= 0):
            raise DupireSurfaceError("Implied volatility requires T>0.")
        w, _, _, _ = self.components(ta, ka, allow_k_clamp=False)
        return np.sqrt(w / ta)

    def bumped(self, *, parallel=0.0, skew=0.0):
        """Bump IMPLIED vols at input nodes and recalibrate Dupire from scratch.

        delta_IV = parallel - skew*tanh(2*k), all in decimal volatility.
        This is an explicit market-surface scenario, different from applying
        the paper's proprietary parameter shift or a fixed local-vol bump.
        """
        parallel = float(parallel)
        skew = float(skew)
        if not np.isfinite(parallel + skew):
            raise DupireSurfaceError("Nonfinite bump.")
        delta = parallel - skew * np.tanh(2 * self.log_moneyness)
        vols = self.implied_vols + delta[None, :]
        provenance = {
            "kind": "recalibrated_implied_vol_bump",
            "parent": self.source,
            "parallel_decimal": parallel,
            "skew_decimal": skew,
            "definition": "delta_IV=parallel-skew*tanh(2*k)",
        }
        return type(self)(
            self.maturities,
            self.log_moneyness,
            vols,
            source=provenance,
            spot=self.spot,
            denominator_floor=self.denominator_floor,
        )

    def validate_dense(self):
        """Finite-domain diagnostics, NOT a global no-arbitrage guarantee."""
        ts = np.unique(
            np.r_[
                0.0,
                self.maturities,
                np.geomspace(max(self.tmin * 1e-4, 1e-8), self.tmin, 24),
                np.linspace(self.tmin, self.tmax, 241),
            ]
        )
        ks = np.unique(np.r_[self.log_moneyness, np.linspace(self.kmin, self.kmax, 401)])
        tt, kk = np.meshgrid(ts, ks, indexing="ij")
        var, g, wt = self.local_variance(tt, kk, allow_k_clamp=False, check=False)
        bad = (
            (~np.isfinite(var))
            | (~np.isfinite(g))
            | (wt <= 0)
            | (g <= self.denominator_floor)
            | (var <= 0)
        )
        if np.any(bad):
            i, j = np.argwhere(bad)[0]
            raise DupireSurfaceError(
                f"Dupire check failed at T={ts[i]:.8g}, k={ks[j]:.8g}: w_T={wt[i,j]:.8g}, g={g[i,j]:.8g}. No automatic repair is applied."
            )
        fitted = self.spline(self.maturities, self.log_moneyness)
        return {
            "status": "passed_finite_dense_grid_not_global_proof",
            "tested_points": int(tt.size),
            "min_calendar_derivative_wT": float(wt.min()),
            "min_butterfly_denominator": float(g.min()),
            "max_butterfly_denominator": float(g.max()),
            "min_local_vol": float(np.sqrt(var.min())),
            "max_local_vol": float(np.sqrt(var.max())),
            "max_total_variance_node_interpolation_error": float(
                np.max(np.abs(fitted - self.total_variances))
            ),
        }

    def metadata(self):
        """Describe the input quotes, interpolation and surface validation for export."""
        return {
            "model": "Dupire local volatility, not stochastic local volatility",
            "source": self.source,
            "spot": self.spot,
            "rates_assumption": "r=q=0",
            "surface_coordinates": "T, k=ln(K/F(T)); here F(T)=spot because r=q=0",
            "interpolator": "RectBivariateSpline cubic total variance, s=0; not a guaranteed arbitrage-free fit",
            "fixed_time_evaluation": "Exact tensor contraction to univariate B-splines; LRU2048; no extra spatial interpolation",
            "maturity_range": [self.tmin, self.tmax],
            "log_moneyness_range": [self.kmin, self.kmax],
            "normalized_spot_range": [
                float(100 * np.exp(self.kmin)),
                float(100 * np.exp(self.kmax)),
            ],
            "nodes": [len(self.maturities), len(self.log_moneyness)],
            "boundary_policy": {
                "short_time": "w(t,k)=t*w(Tmin,k)/Tmin; exact local-vol t=0 limit",
                "long_time": "reject t>Tmax",
                "wings": "freeze LOCAL vol at nearest log-moneyness boundary",
            },
            "denominator_floor_rejection_not_clipping": self.denominator_floor,
            "validation": self.validation,
            "caveat": "Dense-grid checks do not prove global arbitrage absence; boundary continuation may alter repricing and must be tested.",
        }


def write_demo_surface(path, spot=100.0):
    """Write SYNTHETIC SSVI quotes; no historic Eurostoxx calibration implied.

    theta(T)=.04*T+.02*(1-exp(-T)), rho=-.45, phi=.8. Constant phi and
    increasing theta give calendar monotonicity; sufficient SSVI butterfly
    bounds theta*phi*(1+|rho|)<4 and theta*phi²*(1+|rho|)<=4 hold on0<T<=5.
    The input SSVI formula obeys these bounds on that horizon; the subsequent
    spline is separately checked numerically, not assumed arbitrage-free.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = np.array(
        [
            0.01,
            0.02,
            0.04,
            0.08,
            0.125,
            0.25,
            0.375,
            0.5,
            0.75,
            1,
            1.5,
            2,
            2.5,
            3,
            3.5,
            4,
            4.5,
            5,
        ]
    )
    ks = np.linspace(-3, 3, 121)
    rho = -0.45
    phi = 0.8
    theta = 0.04 * ts + 0.02 * (-np.expm1(-ts))
    h = 1 + rho * phi * ks + np.sqrt((phi * ks + rho) ** 2 + 1 - rho * rho)
    w = 0.5 * theta[:, None] * h[None, :]
    vols = np.sqrt(w / ts[:, None])
    bounds = [
        float(theta[-1] * phi * (1 + abs(rho))),
        float(theta[-1] * phi * phi * (1 + abs(rho))),
    ]
    if not bounds[0] < 4 or not bounds[1] <= 4:
        raise AssertionError("SSVI example violates its stated sufficient bounds")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["maturity", "strike", "implied_vol"])
        for i, t in enumerate(ts):
            for j, k in enumerate(ks):
                writer.writerow([f"{t:.15g}", f"{spot*np.exp(k):.15g}", f"{vols[i,j]:.15g}"])
    return {
        "kind": "synthetic_SSVI_example_not_market_data",
        "file_name": path.name,
        "theta": "0.04*T+0.02*(1-exp(-T))",
        "rho": rho,
        "phi": phi,
        "spot": spot,
        "SSVI_sufficient_butterfly_bound_values_at_T5": bounds,
        "bounds_apply_to": "analytic source smile on0<T<=5, not automatically to its interpolating spline",
    }


def validate_dupire_fragment():
    """Independent limiting case and deliberate-invalid-surface tests."""
    import tempfile

    ts = np.array([0.01, 0.25, 1, 2, 5])
    ks = np.linspace(-3, 3, 61)
    flat = DupireLocalVol(ts, ks, np.full((len(ts), len(ks)), 0.3))
    t = np.array([0, 0.001, 0.01, 0.15, 1.7, 5])
    s = np.array([0, 5, 70, 100, 200, 2000])
    maxerr = float(np.max(np.abs(flat(t[:, None], s[None, :]) - 0.3)))
    assert maxerr < 1e-10, maxerr
    a = 0.04
    b = 0.006
    term = DupireLocalVol(
        ts, ks, np.broadcast_to(np.sqrt(a + b * ts[:, None]), (len(ts), len(ks)))
    )
    test_t = np.array([0.05, 0.1, 0.3, 0.8, 1.7, 3.5, 4.8])
    expected = np.sqrt(a + 2 * b * test_t)
    termerr = float(np.max(np.abs(term(test_t, 100) - expected)))
    assert termerr < 1e-10, termerr
    bad = np.full((len(ts), len(ks)), 0.3)
    bad[-1] = 0.05
    rejected = False
    try:
        DupireLocalVol(ts, ks, bad)
    except DupireSurfaceError:
        rejected = True
    assert rejected
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "synthetic.csv"
        source = write_demo_surface(p)
        demo = DupireLocalVol.from_csv(p)
        p1 = demo.bumped(parallel=0.01)
        p2 = demo.bumped(skew=0.01)
        results = {
            "flat_vol_max_abs_error": maxerr,
            "term_structure_max_abs_error": termerr,
            "invalid_calendar_rejected": rejected,
            "demo": demo.metadata(),
            "parallel_bump": p1.validation,
            "skew_bump": p2.validation,
            "demo_source": source,
        }
    return results


def run_replication(
    output_dir,
    *,
    section="all",
    config=None,
    pdf=None,
    vol_surface=None,
    surface_spot=100.0,
    deformation_space="local",
    use_proxy=False,
    repricing_tolerance_bp=5.0,
    validate=True,
):
    """Run Section 3 (theory), Section 4 (fhc), or both (all) and save their outputs.

    config controls the PDE mesh and Monte Carlo sample sizes. For Section 4,
    vol_surface can supply implied quotes in place of the synthetic SSVI quotes.
    Passing pdf also extracts the original figures. For independent checks or
    extraction alone, call run_validation or extract_original_figures directly.
    """
    if section not in ("theory", "fhc", "all"):
        raise ValueError("Choose section='theory', 'fhc' or 'all'.")
    if use_proxy and vol_surface is not None:
        raise ValueError("Choose one model: a supplied quote surface or the synthetic proxy.")
    if deformation_space not in ("implied", "local") or (
        use_proxy and deformation_space == "implied"
    ):
        raise ValueError("Choose local or implied bumps. The proxy supports local bumps only.")
    if not np.isfinite(surface_spot) or surface_spot <= 0:
        raise ValueError("surface_spot must be positive.")
    if not np.isfinite(repricing_tolerance_bp) or repricing_tolerance_bp <= 0:
        raise ValueError("repricing_tolerance_bp must be positive.")
    cfg = Config() if config is None else config
    if (
        min(cfg.paths, cfg.profile_paths) < 2
        or cfg.profile_spots < 3
        or cfg.profile_times_year < 1
    ):
        raise ValueError(
            "Use at least 2 paths per run, 3 profile spots and 1 profile date per year."
        )
    if not 0 < cfg.rcond < 1 or not 72 <= cfg.dpi <= 600:
        raise ValueError("Use 0 < rcond < 1 and a figure resolution from 72 to 600 dpi.")
    if (
        cfg.csv_compression not in ("none", "gzip")
        or cfg.trace_runs not in ("main", "all")
        or cfg.coupon_mode not in ("flat", "snowball")
    ):
        raise ValueError("Check csv_compression, trace_runs and coupon_mode in Config.")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if pdf is None:
        for candidate in [
            Path(__file__).parent / "original_paper.pdf",
            Path.cwd() / "THE COST OF VEGA-HEDGING STRUCTURED PRODUCTS.pdf",
        ]:
            if candidate.is_file():
                pdf = candidate
                break
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": cfg.dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    start = time.perf_counter()
    results = {
        "configuration": asdict(cfg),
        "assumptions": ASSUMPTIONS,
        "source": SOURCE_URL,
        "source_date": "2023-08-24",
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    log("PDE/MC algorithmic reproduction - SSRN 4550643")
    log(
        "Figures 10-16 require unavailable proprietary data: the selected model does not automatically recover their historical values."
    )
    if pdf is not None:
        log("[Original figures] Extracting all 16 figures from the supplied PDF")
        results["originals"] = extract_original_figures(pdf, out / "original_figures", cfg.dpi)
    else:
        log(
            "[Original figures] No PDF: numerical calculations are available; Figure 6 and source images require the original PDF"
        )
    if validate:
        results["validation"] = run_validation(out)
    if section in ("all", "theory"):
        results["section3"] = section3(cfg, out / "section3", results_dir=out)
    if section in ("all", "fhc"):
        if use_proxy:
            model = proxy_model()
        else:
            model = build_dupire_market(
                out,
                vol_surface=vol_surface,
                surface_spot=surface_spot,
                deformation_space=deformation_space,
            )
            results["dupire"] = dupire_diagnostics(model, cfg, out, repricing_tolerance_bp)
            results["volatility_selection"] = {
                "model": model["tag"],
                "deformation_space": deformation_space,
                "surface_spot": surface_spot,
            }
        results["section4"] = section4(
            cfg, out / ("section4_" + model["tag"]), model=model, results_dir=out
        )
    results["elapsed_seconds"] = time.perf_counter() - start
    write_json(out / "run_summary.json", results)
    (out / "ASSUMPTIONS_AND_LIMITATIONS.txt").write_text(
        "ASSUMPTIONS, LIMITATIONS AND CONVENTIONS\n\n"
        + "\n".join(f"{i+1}. {x}" for i, x in enumerate(ASSUMPTIONS)),
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.json":
            manifest.append(
                {
                    "file": path.relative_to(out).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(out / "manifest_sha256.json", manifest)
    log(f"Completed in {results['elapsed_seconds']:.1f}s. Results saved in: {out}")
    return results
