# Vega hedging with Bel Hadj Ayed and Loeper

Open `main.py` first. The four branches of `main` show the experiments you can run. The default is a two-call example that goes from PDE sensitivities to an actual hedge.

## First run

Extract the whole ZIP. Open a terminal in the folder containing `main.py` and use Python 3.10 or later. Install the dependencies once in the same Python environment you use to run the script.

```bash
python -m pip install -r requirements.txt
python main.py
```

You can also open `main.py` in your editor and press **Run**. If your system uses `python3`, use it for both terminal commands. Calculations run on the CPU.

## Choose the experiment

Edit the last line of `main.py`. For instance, to compare future hedging costs (FHC), use

```python
main(studio="confronto_fhc", n_percorsi=10_000)
```

| Value of `studio` | What runs | First output to open, under `risultati/` |
| --- | --- | --- |
| `"esempio"` | Build two PDE sensitivity rows and compute the first hedge of a two-year call spread. | `esempio_quadratico/pesi.csv` |
| `"replica"` | Reproduce the Section 3 call spread, digital, barrier and autocall examples. | `replica/call_spread/` for figures and `replica/section3/data/` for numerical grids |
| `"confronto_fhc"` | Compare the paper's minimum-change rule and the quadratic solver with identity metric on common Monte Carlo paths. | `confronto_fhc/fhc_summary.csv` and `confronto_fhc/paired_differences.csv` |
| `"replica_completa"` | Run Sections 3 and 4, including the reference FHC experiment, its figures and CSV exports. | `replica_completa/run_summary.json` |

`n_percorsi` applies to the two Monte Carlo experiments. In the comparison it sets the number of paired paths and the size of the independent pilot. A smaller value gives a quicker test with more sampling uncertainty. The PDE mesh stays the same. The complete replication also simulates the conditional cost profiles and writes large grid and weight files.

## Follow one hedge in the code

In the `"esempio"` branch of `main.py`, the three calls run in this order.

1. **`call_spread_sensitivities(config)`** prices the two-year call spread and its two hedging calls. At time zero, spot 100 and base volatility 30%, it reads the price changes under a parallel bump and an exponential bump. Each row of `M` is one bump, and each column is one hedging call. The corresponding entry of `b` is the spread's price change. The first array axis indexes market states. This example has one state.
2. **`minimum_change_quadratic(M, b, previous, Q=..., tolerance=...)`** computes the residual `r = b − M @ previous` for each state. If its largest absolute component exceeds the threshold, `solve_quadratic_trade` minimizes ½ ΔwᵀQΔw subject to MΔw = r. Otherwise the trade is zero. The function returns the updated holdings, the trade and the decision to rebalance. Here the initial holdings are zero and Q is the identity.
3. **`save_example(...)`** checks the result and writes `pesi.csv` (holdings before, trade, holdings after) and `sensibilita.csv` (target, sensitivity rows, residual after trading). The expected hedge is **long one strike-100 call and short one strike-110 call**.

## Follow the FHC comparison

Read `run_fhc_comparison` in `run_comparison.py`. It builds the synthetic SSVI–Dupire market and specifies the five-year autocall and 14 hedging puts. The paper's proprietary market surface is unavailable.

1. `load_or_solve_grids` solves the base and bumped PDEs, or reads compatible grids already in `risultati/confronto_fhc/pde_cache/`. Keep this directory to reuse grids between runs.
2. `check_simulator` compares the reference and adapted simulator on the same 256 paths. `estimate_qstar` uses an independent pilot to estimate the typical absolute daily change in exotic vega for the linear-impact cost.
3. `simulate_fhc` runs both hedge rules on the same paths for each deformation. At a rebalancing date it interpolates the price changes at the simulated spot, tests the residual and, when required, updates the put holdings. The five cases are the parallel bump, each of the two exponential bumps and each exponential combined with the parallel bump.
4. `path_costs_bp` converts the accumulated traded vega to pathwise costs in basis points for the constant-spread and linear-impact conventions. The driver writes the means, paired differences, initial weights and cost figures. `initial_weights.csv` contains one initial position per put, deformation case and solver.

## Where to read next

| File | Responsibility |
| --- | --- |
| `main.py` | The single entry point. Choose an experiment at the bottom. |
| `esempio_quadratico.py` | Prepare the small PDE example and save its two output tables. |
| `quadratic_hedge.py` | Check the rebalancing threshold and solve the equality-constrained quadratic problem. |
| `vega_hedging_reproduction.py` | Reference implementation. Start with `run_replication`, then follow `section3` or `section4`. |
| `run_comparison.py` | Prepare the common inputs, run both hedge rules and report paired FHC results. |
| `adapted_simulator.py` | Run the daily hedging loop with the selected scenario rows and hedge rule. |
| `test_quadratic_hedge.py` | Nine tests for the solver, its constraints and the rebalancing threshold. |

To run the solver tests from the project folder, use `python -m unittest -v test_quadratic_hedge`. To include the paper's original figures alongside the computed results, place the original PDF next to `main.py` as `original_paper.pdf` and run a replication mode. Figure 6 is a diagram extracted from that PDF.


### Notice: results in terms of FHC are stored in the excel file risultati/comparison_fhc/fhc_summary