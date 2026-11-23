"""Algorithm 2 simulator adapted directly from the unchanged reproduction.

Only three numerical interfaces differ: choose rows of the sensitivity matrix,
select the initial-hedge solver, select the subsequent minimum-change solver.
The FIRST risk row ALWAYS remains the parallel bump for execution-cost vega,
even when the hedge constraint is exponential only. All Monte Carlo dynamics,
trigger, early-call unwind and maturity conventions are inherited unchanged.
"""

import numpy as np
from vega_hedging_reproduction import minimum_change, proxy_sigma
from quadratic_hedge import solve_quadratic_trade, minimum_change_quadratic


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
    risk_indices=None,
    hedge_solver="loeper",
    Q=None,
):
    """Simulate the five-year autocall and accumulate traded vega along each path.

    risk_indices selects the constraints. hedge_solver selects the rebalancing rule.
    The returned q and q2 are sums of daily traded vega and of its square.
    The cost conversion into basis points takes place in run_comparison.py.
    """
    if hedge_solver not in ("loeper", "quadratic"):
        raise ValueError("Choose hedge_solver='loeper' or 'quadratic'.")
    risk_indices = list(range(factors)) if risk_indices is None else list(risk_indices)
    sigma = proxy_sigma if sigma is None else sigma
    n_paths = cfg.paths if npaths is None else npaths
    seed = cfg.seed if seed is None else seed
    if not 0 <= start_day <= 1260:
        raise ValueError("start_day is outside [0,1260]")
    if period < 1:
        raise ValueError("period must be positive")
    t0 = start_day / 252
    horizon = 5 - t0
    s = np.full(n_paths, spot, dtype=float)
    alive = np.ones(n_paths, dtype=bool)
    duration = np.full(n_paths, horizon)
    total = np.zeros(n_paths)
    squared = np.zeros(n_paths)
    rehedges = np.zeros(n_paths, dtype=int)
    weights = np.zeros((n_paths, 14))
    if start_day == 1260:
        if trace is not None:
            trace.emit(
                np.arange(n_paths),
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
                weights,
                np.zeros_like(weights),
            )
        return dict(
            q=total,
            q2=squared,
            duration=np.zeros(n_paths),
            rehedges=rehedges,
            qstar=None,
            outside_count=0,
            post_residual_max=0.0,
            threshold_breaches_after_update=0,
        )
    initial_sensitivities = risk.sample(t0, [spot])[0]
    A0 = initial_sensitivities[risk_indices, 1:]
    b0 = initial_sensitivities[risk_indices, 0]
    weights[:] = (
        np.linalg.pinv(A0, rcond=cfg.rcond) @ b0
        if hedge_solver == "loeper"
        else solve_quadratic_trade(A0, b0, Q, rcond=cfg.rcond)[0]
    )
    if trace is not None:
        trace.emit(
            np.arange(n_paths),
            start_day,
            s,
            0,
            1,
            1,
            0.0,
            0.0,
            total,
            squared,
            rehedges,
            weights,
            weights,
        )
    previous_exotic_vega = np.full(n_paths, initial_sensitivities[0, 0] / 0.01)
    rng = np.random.default_rng(seed)
    vega_change_sum = 0.0
    vega_change_count = 0
    outside = 0
    max_residual = 0.0
    unsatisfied = 0
    dt = 1 / 252
    sqrt_dt = np.sqrt(dt)
    for day in range(start_day + 1, 1261):
        if not np.any(alive):
            break
        # Draw for ALL paths on ALL business days: common seed really gives
        # common Brownian paths across different hedging periods.
        local_vol = sigma((day - 1) / 252, s)
        s *= np.exp(
            -0.5 * local_vol**2 * dt + local_vol * sqrt_dt * rng.standard_normal(n_paths)
        )
        was_alive = alive.copy()
        annual = day % 252 == 0
        called = alive & (s >= 100) if annual and day < 1260 else np.zeros(n_paths, dtype=bool)
        expired = alive.copy() if day == 1260 else np.zeros(n_paths, dtype=bool)
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
                    np.zeros_like(weights[expiry_ids]),
                    -weights[expiry_ids],
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
                        weights[no_unwind_ids],
                        np.zeros_like(weights[no_unwind_ids]),
                    )
        rebalance = (day - start_day) % period == 0
        needs_sensitivities = (
            was_alive
            if calibrate
            else (called if cfg.include_unwind else np.zeros(n_paths, dtype=bool))
            | (alive if rebalance else np.zeros(n_paths, dtype=bool))
        )
        ids = np.flatnonzero(needs_sensitivities)
        if not ids.size:
            continue
        sensitivities = risk.sample(day / 252, s[ids])
        outside_grid = s[ids] > risk.s[-1]
        outside += int(outside_grid.sum())
        sensitivities[outside_grid] = 0
        vega = np.abs(sensitivities[:, 0, 1:] / 0.01)
        if cfg.include_unwind:
            called_mask = called[ids]
            if np.any(called_mask):
                if trace is not None:
                    unwind_delta = -weights[ids[called_mask]].copy()
                q = np.sum(np.abs(weights[ids[called_mask]]) * vega[called_mask], axis=1)
                total[ids[called_mask]] += q
                squared[ids[called_mask]] += q * q
                weights[ids[called_mask]] = 0
                if trace is not None:
                    path_ids = ids[called_mask]
                    trace.emit(
                        path_ids,
                        day,
                        s[path_ids],
                        2,
                        0,
                        0,
                        q,
                        q * q,
                        total[path_ids],
                        squared[path_ids],
                        rehedges[path_ids],
                        weights[path_ids],
                        unwind_delta,
                    )
        if rebalance:
            alive_mask = alive[ids]
            if np.any(alive_mask):
                path_ids = ids[alive_mask]
                M = sensitivities[alive_mask][:, risk_indices, 1:]
                target = sensitivities[alive_mask][:, risk_indices, 0]
                if hedge_solver == "loeper":
                    new_weights, trade, traded = minimum_change(
                        M, target, weights[path_ids], cfg.rcond, cfg.loss_tolerance
                    )
                else:
                    new_weights, trade, traded = minimum_change_quadratic(
                        M,
                        target,
                        weights[path_ids],
                        Q=Q,
                        rcond=cfg.rcond,
                        tolerance=cfg.loss_tolerance,
                    )
                q = np.sum(np.abs(trade) * vega[alive_mask], axis=1)
                total[path_ids] += q
                squared[path_ids] += q * q
                rehedges[path_ids] += traded
                weights[path_ids] = new_weights
                if np.any(traded):
                    residual = target[traded] - np.einsum(
                        "nij,nj->ni", M[traded], new_weights[traded]
                    )
                    peak = np.max(np.abs(residual), axis=1)
                    max_residual = max(max_residual, float(peak.max()))
                    unsatisfied += int(np.sum(peak > cfg.loss_tolerance))
                    if trace is not None:
                        updated = path_ids[traded]
                        trace.emit(
                            updated,
                            day,
                            s[updated],
                            1,
                            1,
                            1,
                            q[traded],
                            q[traded] * q[traded],
                            total[updated],
                            squared[updated],
                            rehedges[updated],
                            new_weights[traded],
                            trade[traded],
                        )
        if calibrate:
            current_exotic_vega = sensitivities[:, 0, 0] / 0.01
            current_exotic_vega[ending[ids]] = 0
            vega_change_sum += float(
                np.sum(np.abs(current_exotic_vega - previous_exotic_vega[ids]))
            )
            vega_change_count += len(ids)
            previous_exotic_vega[ids] = current_exotic_vega
    return dict(
        q=total,
        q2=squared,
        duration=duration,
        rehedges=rehedges,
        qstar=vega_change_sum / vega_change_count if vega_change_count else None,
        outside_count=outside,
        post_residual_max=max_residual,
        threshold_breaches_after_update=unsatisfied,
    )
