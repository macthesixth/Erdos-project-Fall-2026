# LOCAL PYTHON — VARIATIONAL GBM HEDGING

This folder is designed to run directly on Windows, macOS, or Linux.
It does not need Google Colab or Jupyter. The single `.py` file contains
both your original GBM Method-2 functions (normalizing flow, training,
control, EM, PDE reference) and the new synthetic hedging experiment.

## INSTALL (ONE TIME)

Open Terminal / PowerShell in this folder and run:

    python -m pip install -r requirements.txt

If your system invokes Python as `py` or `python3`, use that command instead.
On Windows, `py -m pip install -r requirements.txt` is often appropriate.

## RUN (QUICK VERIFICATION)

    python GBM_Variational_Hedging_Local.py

This uses 1 path, 2 observation dates, 40 optimization steps per observation
prefix. It is just a SMOKE TEST and produces results in `results_local/`.

## PILOT EXPERIMENT (ACTUAL NEURAL METHOD 2)

    python GBM_Variational_Hedging_Local.py --paths 8 --obs 3 --train-steps 200 --samples 1024 --particles 12000 --outdir pilot_results

## STRONGER EXPERIMENT (SLOWER)

    python GBM_Variational_Hedging_Local.py --paths 30 --obs 3 --train-steps 350 --samples 2048 --particles 12000 --outdir study_results

You can set any of `--paths`, `--obs`, `--train-steps`, `--samples`,
`--particles`, `--R` (price-observation noise variance), `--cost`
(proportional transaction cost rate), `--seed`, and `--outdir`.
See:

    python GBM_Variational_Hedging_Local.py --help

## WHAT THE CODE DOES

1. Simulates exact latent GBM paths with known physical drift and volatility.
2. Generates synthetic *price-space* noisy observations.
3. Runs a bootstrap particle filter as an independent posterior benchmark.
4. Trains your original neural Method-2 flow on observation prefixes, so
   the hedge at t_k never uses observations after t_k.
5. Computes posterior plug-in and posterior-averaged Black-Scholes deltas.
6. Tests regular and uncertainty-adaptive delta hedging with transaction costs.
7. Writes CSV tables and PNG figures to the requested output directory.

Compared strategies: Noisy BS; PF plug-in; PF averaged; VI plug-in;
VI averaged; VI adaptive; Oracle diagnostic. Oracle uses hidden latent
prices and must NOT be described as implementable in real life.

## OUTPUT FILES

- `summary.csv`: RMSE and CVaR of terminal hedge losses, trading costs, count.
- `per_path_hedging.csv`: per-path hedging P&L and costs for each strategy.
- `posterior_diagnostics.csv`: variational and particle-filter estimates.
- `latent_vs_filtered.png`: path and posterior-mean comparison.
- `target_deltas.png`: target hedge deltas and adaptive holdings.
- `hedge_rmse.png`: hedging error comparison.
- `experiment_config.json`: all experiment inputs and simulation caveat.

## IMPORTANT DETAILS

- Model and code use S0=K=1, mu=0.08, sigma=0.1, r=0.03, T=0.8;
  observation Y=S+epsilon with epsilon~N(0,R) and R=0.0009 by default.
- Black-Scholes valuation uses risk-free r, NOT physical mu.
- Training many paths with repeated prefix-flow fitting is computationally
  expensive; start with the quick run, then pilot if the quick run succeeds.
- Short-run variational estimates may be inaccurate due to undertraining.
- Synthetic execution price is the simulated latent price; the hedge selection
  algorithm is not shown that price. This separates observation quality from
  microstructure and execution effects.
- This is a synthetic validation experiment, NOT evidence of live-trading P&L.
- The old `run_em` functions remain in the source but are not called when
  you run the hedging experiment; they are optional for separate research.

## IF PACKAGES FAIL TO INSTALL

Install a current supported version of Python from https://www.python.org/.
For PyTorch platform-specific installation information, see
https://pytorch.org/get-started/locally/ .
