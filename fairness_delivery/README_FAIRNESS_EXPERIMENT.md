Relay-Burden Fairness Experiment — Integration Notes
=======================================================

This delivers the code and raw results behind Section 7.8 (Relay-Burden
Fairness) of the paper. It's built to drop directly into your existing
neet-wsn-routing repository, reusing its src/ library unchanged.

## Where these files go

Given your existing repository layout:

    neet-wsn-routing/
      src/            <- already has network.py, flow_controller.py,
                          flow_controller_mincost.py, metrics.py, etc.
      experiments/    <- put fairness_sweep_lib.py and run_fairness.py here
      results/        <- put fairness_results/ (the whole folder) here

Copy this bundle's experiments/*.py into your repo's experiments/ folder,
and this bundle's results/fairness_results/ into your repo's results/
folder. No changes to src/ are needed or made — this experiment only reads
from it.

## What's in it

- fairness_sweep_lib.py — extends the existing, verified NEET/NEET-Cost
  episode loop (same solve_flow_lp / solve_flow_lp_mincost solvers used
  throughout the rest of the repo) to additionally track each node's
  cumulative carried traffic (own + relayed) across its alive lifetime,
  and compute Jain's fairness index over it at episode end. Not a new
  simulator — a targeted extension of the existing one.

- run_fairness.py — the resumable runner script. Usage:

      cd experiments/
      python3 run_fairness.py neet 0 10       # NEET, seeds 0-9
      python3 run_fairness.py neet_cost 0 10  # NEET-Cost, seeds 0-9

  Each (objective, seed) result is cached to results/fairness_results/
  immediately and skipped on rerun — safe to interrupt and resume, same
  pattern as every other experiment script in this repository.

- results/fairness_results/*.json — the 20 raw results (10 seeds x 2
  configurations) that produced Table 5 in the paper: Jain's index
  NEET=0.876 (σ=0.022), NEET-Cost=0.842 (σ=0.038), Mann-Whitney p=0.045,
  Cohen's d=1.05.

## Correctness check already performed

Before trusting any fairness number, the underlying HND values from this
script were cross-checked against the paper's own verified results:
NEET N=100 mean HND=631.7 here vs. 620.7 in the paper; NEET-Cost mean
HND=730.5 here vs. 718.4 in the paper — both within normal seed-to-seed
variation (this run uses a different, smaller n=10 sample than the
paper's primary n=15 tables), confirming this extension faithfully
reproduces the same underlying dynamics before any new metric from it
was trusted.
