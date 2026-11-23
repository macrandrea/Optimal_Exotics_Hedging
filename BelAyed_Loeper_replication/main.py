"""Edit the call at the bottom, then press Run or type: python main.py."""

from pathlib import Path

import numpy as np

from esempio_quadratico import call_spread_sensitivities, save_example
from quadratic_hedge import minimum_change_quadratic
from run_comparison import run_fhc_comparison
from vega_hedging_reproduction import Config, run_replication

RESULTS = Path(__file__).resolve().parent / "risultati"

#%%
def main(studio="example", n_paths_mc=10_000):
    """Choose the experiment. n_paths_mc is used only by the Monte Carlo runs."""
    if studio == "example":
        # 1. Read M (call price changes) and b (spread price changes) from the PDEs.
        config = Config()
        M, b = call_spread_sensitivities(config)

        # 2. Start with zero holdings. If the residual exceeds the threshold,
        #    find the smallest trade that matches b, using Q = identity.
        previous = np.zeros((1, 2))  # One market state, two hedging calls.
        weights, trade, rebalanced = minimum_change_quadratic(
            M, b, previous, Q=np.eye(2), tolerance=config.loss_tolerance
        )

        # 3. Print the hedge and save the sensitivities and positions as CSVs.
        print(f"Ribilanciamento attivo: {bool(rebalanced[0])}")
        save_example(RESULTS / "example_quadratico", M, b, previous, weights, trade)

    elif studio == "replica":
        # Section 3: reproduce the theoretical examples, figures and PDE data.
        run_replication(RESULTS / "replica", section="theory")

    elif studio == "comparison_fhc":
        # Price the autocall and 14 puts, calibrate the cost on a pilot sample,
        # then compare the two hedge rules on the same Monte Carlo paths.
        run_fhc_comparison(RESULTS / "comparison_fhc", paths=n_paths_mc)

    elif studio == "complete_replication":
        # Sections 3 and 4: include the original FHC experiment on synthetic data.
        config = Config(paths=n_paths_mc, profile_paths=n_paths_mc, csv_compression="gzip")
        run_replication(RESULTS / "complete_replication", section="all", config=config)

    else:
        raise ValueError("Choose 'example', 'replica', 'comparison_fhc' or 'complete_replication'.")

#%%
#if __name__ == "__main__":
    # Choose here: "example", "replica", "comparison_fhc" or "complete_replication".
    #main(studio="complete_replication", n_paths_mc=10_000)
#%%
if __name__ == "__main__":
    # Choose here: "example", "replica", "comparison_fhc" or "complete_replication".
    main(studio="comparison_fhc", n_paths_mc=10_000)