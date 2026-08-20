# NEET: Multi-Path Flow Optimization for WSN Routing

Experimental pipeline supporting *"NEET: A Multi-Path Flow Optimization
Framework for Wireless Sensor Network Lifetime Maximization — Theory, the
Fairness–Efficiency Tradeoff, and Empirical Characterization"* (submitted to
ACM Transactions on Sensor Networks).

This is a real, working research pipeline built and run incrementally over
the course of the investigation — not a polished library. The repository is
organized to separate the reusable core model (`src/`), the scripts that
actually produced the paper's reported numbers (`experiments/`), the raw
result data those scripts generated (`results/`), and the full diagnostic
history of the investigation (`archive/`), preserved for transparency and
reproducibility rather than tidiness.

## Repository structure

```
src/                   Core, reusable library modules
experiments/           Final experiment-runner scripts, each mapped below
                        to the specific paper section/table/figure it produced
results/               Raw JSON result files, one subdirectory per experiment
figures/               The final PDF figures used in the paper
archive/diagnostics/   Exploratory, calibration, and debugging scripts from
                        the investigation process (kept for transparency —
                        not part of the "clean path" to reproduce results)
archive/logs/          Captured stdout logs from long-running experiment
                        scripts (useful for auditing exact run parameters
                        and timings)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. The LP solver used throughout (`src/flow_controller*.py`)
is `scipy.optimize.linprog` with `method='highs'` — no external solver
installation needed.

## Core library (`src/`)

| File | Purpose |
|---|---|
| `network.py` | Node deployment, communication graph construction |
| `energy.py` | First-order radio energy model (transmit/receive cost) |
| `metrics.py` | Ψ, Ψ_N, structural covariance Λ, and the risk indicator R_h |
| `controller.py`, `simulate.py` | Legacy v2-era single-path tree controllers (`static`, `mincost`/`greedy-cost-tree`, `eqopt`) and episode loop — these are the tree-routing baselines Theorem 2 is validated against (paper Section 7.1) |
| `flow_controller.py`, `flow_simulate.py` | **NEET**: the min-max relative-depletion multi-path flow LP and its episode loop |
| `flow_controller_mincost.py`, `flow_simulate_mincost.py` | **NEET-Cost**: the total-cost-minimizing multi-path flow LP and its episode loop |
| `flow_controller_blend.py`, `flow_simulate_blend.py` | The blended objective J_α interpolating between NEET and NEET-Cost (paper Section 8) |
| `flow_simulate_static_ct.py` | Static Chang–Tassiulas baseline: one-shot LP solve, no re-optimization (paper Section 5.1) |
| `flow_controller_agg.py`, `flow_simulate_agg.py` | In-network aggregation-discount variant tested and retracted during the root-cause investigation (paper Section 7.2, mechanism 5) |
| `protocols.py` | LEACH, HEED, PEGASIS baselines, corrected for the epoch-tracking defect (paper Section 6.3) |
| `protocol_eeuc.py` | EEUC modern baseline, corrected for both the epoch-tracking and relay-selection defects (paper Section 6.3) |
| `stats_utils.py` | Mann–Whitney U, Cohen's d, Spearman correlation helpers |
| `plotting.py` | Shared figure-generation utilities |

## Reproducing the paper's results

| Paper result | Script | Result data |
|---|---|---|
| Core scaling validation vs. tree baselines (§7.1, Fig. 1) | `experiments/n_sweep_v3_vs_protocols.py`, `experiments/n_sweep_v3_extended.py` | `results/n_sweep_v3_vs_protocols_results/`, `results/n_sweep_v3_results/` |
| NEET-Cost vs. static Chang–Tassiulas (§7.3, Table 1, Fig. 2) | `experiments/final_strategy_sweep.py` | `results/final_strategy_results/` |
| NEET-Cost vs. LEACH/HEED, corrected (§7.4, Table 2, Fig. 3) | `experiments/redo_leach_comparisons.py` | `results/redo_leach_results/` |
| EEUC full validation vs. all baselines (§7.5, Table 3, Fig. 4) | `experiments/final_remaining_experiments.py` | `results/final_remaining_results/` |
| Energy-budget and sink-topology sensitivity (§7.6, Fig. 5) | `experiments/final_remaining_experiments.py` (Priorities 2–3) | `results/final_remaining_results/` |
| Fair hole-risk comparison (§7.7, Fig. 6) | `experiments/holerisk_leach_heed.py` | `results/holerisk_leach_heed_results/` |
| Blended-objective tradeoff, N=500 completion (§8.3) | `experiments/final_remaining_experiments.py` (Priority 4) | `results/final_remaining_results/blend_N500_alpha*.json` |
| mincost-objective correctness validation | `experiments/validate_mincost_objective.py` | — |

> **Known gap:** the N=100 blend sweep (`α ∈ {0, 0.05, ..., 1.0}`, n=10) that
> produced the paper's Figure 7 main result is **not present in this
> repository snapshot** — neither the script nor its raw output JSON. The
> `overnight_remaining_experiments.py` script's own comments reference "the
> N=100 version" as a prior, separate script, confirming one existed at some
> point but was not captured in this upload. Locate and add it (or
> regenerate it from `src/flow_controller_blend.py` /
> `src/flow_simulate_blend.py`, following the same pattern as the N=500
> completion in `final_remaining_experiments.py`) before treating this
> repository as a complete reproduction package.

Every experiment script writes its results to disk immediately and is safe
to interrupt and resume — completed cells are detected and skipped on rerun.
Several of the longer sweeps (`final_remaining_experiments.py` in particular)
are genuinely multi-hour jobs at N=500; running with `caffeinate` (macOS) or
equivalent is recommended to prevent sleep interruption.

`results/v2_legacy_results/` and the `eqopt_h`/`eqopt_k`-related scripts in
`archive/diagnostics/` (`alpha_sweep_N100.py`, `screen_across_N.py`) are
from an earlier project stage (the v2 single-source tree-routing model) and
are unrelated to the current paper's NEET/NEET-Cost results — kept for
provenance, not part of the reproduction path above.

## `archive/diagnostics/`: the investigation record

This paper's root-cause investigation (§7.2) tested seven candidate
mechanisms before identifying the true cause of NEET's underperformance
against clustering baselines at scale, and two genuine implementation
defects (in LEACH and EEUC respectively) were found and corrected along the
way (§6.3). The 57 scripts in `archive/diagnostics/` are the actual record
of that process — calibration sweeps, root-cause checks, hypothesis tests
that were tried and ruled out, and defect-diagnosis scripts. They are
preserved deliberately rather than deleted, consistent with this paper's
overall commitment to methodological transparency, but they are **not**
the recommended entry point for reproducing the paper's reported results —
use `experiments/` for that.

## Citation

If you use this code, please cite:

```bibtex
@article{batri2026neet,
  author  = {Batri, Krishnan and Geetha, G. and Lakshmi, S. and Bhatia Khan, Surbhi},
  title   = {NEET: A Multi-Path Flow Optimization Framework for Wireless
             Sensor Network Lifetime Maximization},
  journal = {ACM Transactions on Sensor Networks},
  year    = {2026},
  note    = {In submission}
}
```
