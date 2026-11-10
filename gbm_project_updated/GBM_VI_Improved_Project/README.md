# GBM Variational Posterior Hedging — Revised Local Project

This project is a standalone Python program for GitHub Codespaces, Linux, macOS,
and Windows. It uses synthetic GBM data with **additive Gaussian noise in stock
prices**, not log-prices.

## What changed compared with your original implementation?

### 1. First-observation variance calibration (new, optional)

The uploaded 30-path experiment indicated that the original Method 2 flow
underestimated the FIRST posterior stock-price standard deviation. Simply
training longer did not resolve this on the representative path.

For the **first observation only**, the exact Bayesian posterior is numerically
tractable because the prior to that observation is a known lognormal GBM law:

    prior(log S_t) = Normal(log(S0) + (mu - sigma^2/2)t, sigma^2 t)
    likelihood(Y_t | S_t=s) = Normal(Y_t; s, R)

The script computes its first two **log-price moments** by deterministic
Gauss-Hermite quadrature. It adds a differentiable moment-matching term to the
original flow loss, with a gradual weight ramp to avoid unstable optimization.

**Important:** This is a **hybrid, moment-supervised Method 2**, not the original
unmodified variational objective. It is designed to correct the **first**
causal filtering marginal, not automatically guarantee calibrated distributions
at later dates. Run with `--first-moment-weight 0` to use the unchanged original
Method 2 objective, for a fair ablation.

### 2. Risk-weighted posterior delta (new strategy)

For an option delta D(S), the conditional instantaneous diffusion error under
GBM is proportional to S*(D(S)-h). Minimizing its conditional mean-square yields:

    h* = E_q[S^2 * D(S)] / E_q[S^2].

The script evaluates this with flow samples. This is an instantaneous quadratic
risk criterion, **not** a guarantee of lower total hedging P&L RMSE.

Additional methods in `summary.csv` are:

- `PF risk-weighted` (particle filter benchmark)
- `VI risk-weighted` (your neural posterior)
- `VI risk-weighted adaptive` (risk-weighted target plus no-trade threshold)

Existing methods (`Noisy BS`, `VI averaged`, `VI adaptive`, etc.) are preserved.

### 3. New diagnostics and no-retraining scenarios

New files:

- `first_step_posterior.png`: original neural-posterior samples versus exact
  first-step Bayesian filtering density (same observed price).
- `delta_diagnostics.csv`: true-price/oracle delta and all estimated deltas.
- `hedge_cache.npz`: per-path execution prices, hedge targets, and posterior
  uncertainty, sufficient to **rescore costs and thresholds without retraining**.
- `risk_cost_frontier.png`: exploratory adaptive trade-off on existing paths.
- `first_prefix_validation.csv` + plot: compare baseline and calibrated posterior
  SD on your old experimental observations.
- Existing hedging CSV summaries and PNG plots continue to be generated.

## Installation (GitHub Codespaces)

In the project folder, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1` instead.

## First check — small, fast run

```bash
python GBM_Variational_Hedging_Revised.py \
  --paths 1 --obs 3 --train-steps 60 --samples 512 \
  --particles 3000 --outdir smoke_results
```

**A 60-step smoke run proves only that the program executes.** Posterior means
and variances can be inaccurate at this training length.

## First scientific check — first-time calibration on your old results

Put your previous `posterior_diagnostics.csv` in the same directory, and run:

```bash
python GBM_Variational_Hedging_Revised.py \
  --validate-first-prefix posterior_diagnostics.csv \
  --validation-count 3 --train-steps 350 --samples 4096 \
  --R 0.0009 --outdir first_prefix_validation
```

This reads the observed prices from your previous experiment and checks the
**earliest observation** for three paths. Repeat with `--first-moment-weight 0`
to verify that the original (uncalibrated) implementation is preserved.

## Run the updated hedging comparison

```bash
python GBM_Variational_Hedging_Revised.py \
  --paths 8 --obs 3 --train-steps 350 --samples 2048 \
  --particles 12000 --seed 123 --R .0009 --cost .001 \
  --outdir revised_pilot
```

For an original-Method-2 ablation, run the SAME parameters with
`--first-moment-weight 0 --outdir original_pilot`.

The main model parameters are still normalized S0=K=1, physical mu=.08,
volatility sigma=.1, risk-free r=.03, maturity T=.8. Each path fits one neural
flow per observation prefix. Larger runs can take considerable CPU time.

## Fast transaction-cost / threshold comparisons

After one model-fitting run, do **not** retrain for every threshold:

```bash
python GBM_Variational_Hedging_Revised.py \
  --rescore-cache revised_pilot/hedge_cache.npz \
  --cost .002 --delta0 .01 --lambda-uncertainty .5 \
  --outdir alternative_hedge_settings
```

This changes the costs and adaptive rule on the SAME cached targets. Varying
`--R`, the inference architecture, physical drift, or observations **does**
require a new full inference run.

## What has been tested?

- The revised standalone script ran successfully end-to-end on a small synthetic
  sample, creating the additional VI/PF risk-weighted strategies and all files.
- The no-retraining cache run was checked to reproduce the original P&L,
  transaction costs, and rebalance counts exactly under matching parameters.
- First-observation posterior standard deviations on two paths from the uploaded
  30-path CSV improved from 0.01239 -> 0.02495 and 0.01584 -> 0.02478;
  exact first-step Bayesian references were 0.02517 and 0.02563 respectively.
  A third path tested independently also improved (0.00962 -> 0.02465,
  reference 0.02527).

**Not yet established:** whether the calibration improves hedge P&L on a
large independent test set; whether later-time posteriors are calibrated;
whether risk-weighted delta beats other strategies out of sample. Don't
present small-path tests or in-sample threshold sweeps as proof of superiority.

## Report guidance

Use paired comparisons on the SAME synthetic paths. Report inference quality
(mean error, SD ratio, credible interval coverage) before ranking financial
methods. Separate original Method 2, hybrid first-step-calibrated Method 2,
particle filter, and latent-price oracle. Reserve independent paths for final
hedging evaluation. The synthetic execution model uses the unavailable latent
price for trading, by design: results do not claim real-world tradability.
