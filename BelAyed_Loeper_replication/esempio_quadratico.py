"""From finite-bump PDE sensitivities to the first hedge of a two-year call spread."""

from pathlib import Path
import csv
import numpy as np

import vega_hedging_reproduction as replica


def call_spread_sensitivities(config):
    """Return M (two hedging calls) and b (the spread) for two bumps at t=0, S=100."""

    products = [
        replica.Product("spread", 2, strike=100, strike2=110),
        replica.Product("call", 2, strike=100),
        replica.Product("call", 2, strike=110),
    ]
    sigma = replica.flat_sigma(0.30)
    scenarios = [
        sigma,
        replica.deformation(sigma, 0.003),
        replica.deformation(sigma, 0.003, lam=1.0),
    ]
    _, sensitivity_grid = replica.risk_bundle(products, scenarios, config)

    # One market state, two volatility scenarios, and two hedging calls.
    responses = sensitivity_grid.sample(0.0, [100.0])
    b = responses[:, :, 0]
    M = responses[:, :, 1:]
    return M, b


def save_example(output, M, b, previous, weights, trade):
    """Check the static hedge and save the sensitivities, trades and final holdings."""
    residual = b[0] - M[0] @ weights[0]
    np.testing.assert_allclose(weights[0], [1.0, -1.0], rtol=0, atol=1e-7)
    np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-12)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sensibilita.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["scenario", "target_b", "M_call_K100", "M_call_K110", "post_trade_residual"]
        )
        for i, scenario in enumerate(["parallel", "exponential"]):
            writer.writerow([scenario, b[0, i], *M[0, i], residual[i]])
    with (output / "pesi.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["strike", "previous_holding", "trade", "new_holding"])
        for j, strike in enumerate([100, 110]):
            writer.writerow([strike, previous[0, j], trade[0, j], weights[0, j]])
    for strike, weight in zip([100, 110], weights[0]):
        print(f"Call K={strike}: {weight:+.8f}")
    print(f"Residuo massimo dopo lo scambio: {np.max(np.abs(residual)):.3e}")
    print(f"CSV salvati in {output}")
