"""Equality-constrained minimum-change hedging; no market model is assumed.

Solve min_x 0.5*x.T@Q@x subject to M@x=r. Q must be symmetric positive
definite. Arrays can be batched over states. Rank-deficient, feasible systems
are supported; infeasible constraints raise instead of silently becoming
least-squares hedges. No ridge, position cap, or invented liquidity weights.
"""

from __future__ import annotations
import numpy as np


class InfeasibleHedgeError(ValueError):
    """The requested residual is outside the numerical range of M."""


def solve_quadratic_trade(
    M, r, Q=None, *, rcond=1e-10, feasibility_atol=1e-10, feasibility_rtol=1e-8
):
    """Return (trade, diagnostics), using a linear solve, not an explicit inverse.

    For Q=L L.T, set B=M L^{-T}, z=L.T x. If B.T=U R (thin QR),
    solve R.T y=r, then x=L^{-T} U y. This is algebraically the Schur
    formula Q^{-1} M.T (M Q^{-1} M.T)^{-1} r, but does not square the
    condition number by constructing the Gram matrix. Rank loss is checked
    with the same relative SVD cutoff as the source reproduction.
    """
    M = np.asarray(M, dtype=float)
    r = np.asarray(r, dtype=float)
    single = M.ndim == 2
    if single:
        M = M[None]
        r = r[None]
    if M.ndim != 3 or r.shape != M.shape[:2]:
        raise ValueError("Expected M=(states,risks,instruments), r=(states,risks).")
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(r)):
        raise ValueError("Nonfinite hedge inputs.")
    n, m, d = M.shape
    if min(m, d) < 1 or not 0 < rcond < 1:
        raise ValueError("Invalid dimensions or rcond.")
    if Q is None:
        L = None
        B = M
    else:
        Q = np.asarray(Q, dtype=float)
        if Q.shape != (d, d) or not np.all(np.isfinite(Q)):
            raise ValueError("Q must be a finite instrument-by-instrument matrix.")
        if not np.allclose(Q, Q.T, rtol=1e-12, atol=1e-14):
            raise ValueError("Q must be symmetric.")
        L = np.linalg.cholesky(Q)
        B = np.linalg.solve(L, M.swapaxes(-1, -2)).swapaxes(-1, -2)
    if n == 0:
        return np.empty((0, d)), {
            "constraint_residual": np.empty((0, m)),
            "rank": np.empty(0, dtype=int),
            "condition": np.empty(0),
            "objective": np.empty(0),
        }
    # Solve in whitened coordinates so the objective is ordinary squared length.
    singular_values = np.linalg.svd(B, compute_uv=False)
    keep = singular_values > rcond * singular_values[:, :1]
    rank = keep.sum(axis=1)
    full_row_rank = rank == m
    z = np.zeros((n, d))
    if np.any(full_row_rank):
        U, R = np.linalg.qr(B[full_row_rank].swapaxes(-1, -2), mode="reduced")
        y = np.linalg.solve(R.swapaxes(-1, -2), r[full_row_rank, :, None])[..., 0]
        z[full_row_rank] = np.einsum("nij,nj->ni", U, y)
    # Dependent constraints have a minimum-norm solution when they are feasible.
    if np.any(~full_row_rank):
        z[~full_row_rank] = np.einsum(
            "nji,ni->nj", np.linalg.pinv(B[~full_row_rank], rcond=rcond), r[~full_row_rank]
        )
    trade = z if L is None else np.linalg.solve(L.T, z.T).T
    residual = np.einsum("nij,nj->ni", M, trade) - r
    peak = np.max(np.abs(residual), axis=1)
    allowed = feasibility_atol + feasibility_rtol * np.max(np.abs(r), axis=1)
    bad = peak > allowed
    if np.any(bad):
        ids = np.flatnonzero(bad)
        raise InfeasibleHedgeError(
            f"Infeasible or numerically unresolved constraints at states {ids[:10].tolist()}; "
            f"max absolute residual={peak[bad].max():.6g}. No approximate hedge accepted."
        )
    condition = np.divide(
        singular_values[:, 0],
        singular_values[:, -1],
        out=np.full(n, np.inf),
        where=singular_values[:, -1] > 0,
    )
    info = {
        "constraint_residual": residual,
        "rank": rank,
        "condition": condition,
        "objective": 0.5 * np.sum(z * z, axis=1),
    }
    if single:
        return trade[0], {key: value[0] for key, value in info.items()}
    return trade, info


def minimum_change_quadratic(M, b, previous, *, Q=None, rcond=1e-10, tolerance=1e-5):
    """Return new holdings, trades and the rebalancing mask for a batch of states.

    M[state, scenario, instrument] holds the price change of each hedge instrument.
    b[state, scenario] holds the exotic price change. Rebalance only where the
    largest absolute component of b - M @ previous exceeds tolerance. The
    default 1e-5 represents 0.1 bp of a note with unit notional.
    """
    residual = b - np.einsum("nij,nj->ni", M, previous)
    rebalance = np.max(np.abs(residual), axis=1) > tolerance
    trade = np.zeros_like(previous)
    if np.any(rebalance):
        trade[rebalance], _ = solve_quadratic_trade(
            M[rebalance], residual[rebalance], Q, rcond=rcond
        )
    return previous + trade, trade, rebalance
