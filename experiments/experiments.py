"""
experiments.py -- Experiment A-F runners for the NEET v2 pipeline.

Mirrors the original six-experiment design (Table II of the rejected
manuscript) on top of the rebuilt multi-hop, multi-source model.
Each runner returns a pandas DataFrame of per-run results; summary
tables and stats are computed by run_all.py.
"""
import numpy as np
import pandas as pd
from simulate import run_episode
from protocols import PROTOCOL_FUNCS
import stats_utils as S

STRATEGIES = ['static', 'random', 'greedy', 'mincost', 'eqopt']


def _run_many(N, E0, R_c, sink_pos, strategy, n_runs, max_rounds, seed0=0):
    rows = []
    for r in range(n_runs):
        res = run_episode(N=N, E0=E0, R_c=R_c, sink_pos=sink_pos,
                           strategy=strategy, max_rounds=max_rounds,
                           seed=seed0 + r, record_series=False)
        res['N'] = N; res['E0'] = E0; res['strategy'] = strategy; res['seed'] = seed0 + r
        rows.append(res)
    return rows


# ------------------------------------------------------------------
def experiment_A_scaling(Ns=(30, 60, 100), E0=1.0, R_c=35.0, n_runs=5, max_rounds=3000):
    """Scale invariance: AUC(Psi) ratio should hold roughly constant across N."""
    rows = []
    for N in Ns:
        for strat in STRATEGIES:
            rows += _run_many(N, E0, R_c, None, strat, n_runs, max_rounds)
    return pd.DataFrame(rows)


def experiment_B_energy(E0s=(0.5, 1.0, 2.0), N=60, R_c=35.0, n_runs=5, max_rounds=3000):
    """Energy sensitivity: does EqOpt keep AUC(Psi_N) near-invariant across E0?"""
    rows = []
    for E0 in E0s:
        for strat in STRATEGIES:
            rows += _run_many(N, E0, R_c, None, strat, n_runs, max_rounds)
    return pd.DataFrame(rows)


def experiment_C_topology(N=60, E0=1.0, R_c=35.0, n_runs=5, max_rounds=3000, L=100.0):
    """Sink topology robustness: center / edge / corner."""
    sinks = {'center': [L / 2, L / 2], 'edge': [L, L / 2], 'corner': [L, L]}
    rows = []
    for name, pos in sinks.items():
        for strat in STRATEGIES:
            batch = _run_many(N, E0, R_c, pos, strat, n_runs, max_rounds)
            for b in batch:
                b['sink_topology'] = name
            rows += batch
    return pd.DataFrame(rows)


def experiment_D_correlation(Ns=(30, 60, 100), E0=1.0, R_c=35.0, n_runs=5, max_rounds=3000):
    """Pooled correlation analysis: R_h_mean / AUC(Psi) vs HND/FND, with
    partial correlation controlling for N (reuses Experiment A-shaped data
    if available, but runs independently here for a clean pooled set)."""
    rows = []
    for N in Ns:
        for strat in STRATEGIES:
            rows += _run_many(N, E0, R_c, None, strat, n_runs, max_rounds)
    df = pd.DataFrame(rows)

    results = {}
    for pair_name, xcol, ycol in [('AUC_Psi_vs_HND', 'AUC_Psi', 'HND'),
                                   ('AUC_PsiN_vs_HND', 'AUC_PsiN', 'HND'),
                                   ('Rh_mean_vs_HND', 'Rh_mean', 'HND'),
                                   ('Rh_mean_vs_FND', 'Rh_mean', 'FND')]:
        r_simple, p_simple = S.spearman_simple(df[xcol], df[ycol])
        r_partial, p_partial = S.spearman_partial(df[xcol], df[ycol], df['N'])
        results[pair_name] = dict(rho_simple=r_simple, p_simple=p_simple,
                                   rho_partial=r_partial, p_partial=p_partial)
    return df, results


def experiment_E_ablation(N=60, E0=1.0, R_c=35.0, n_runs=8, max_rounds=3000):
    """Ablation: same as primary comparison, but explicit isolation of
    Psi-driven selection (eqopt) vs the other four heuristics, with paired
    Mann-Whitney / Cohen's d against eqopt."""
    rows = []
    for strat in STRATEGIES:
        rows += _run_many(N, E0, R_c, None, strat, n_runs, max_rounds)
    df = pd.DataFrame(rows)

    stats_out = {}
    eq = df[df.strategy == 'eqopt']
    for strat in STRATEGIES:
        if strat == 'eqopt':
            continue
        other = df[df.strategy == strat]
        for metric in ['AUC_Psi', 'AUC_PsiN', 'HND']:
            u, p = S.mann_whitney(eq[metric], other[metric])
            d = S.cohens_d(eq[metric].values, other[metric].values)
            stats_out[f'eqopt_vs_{strat}_{metric}'] = dict(U=u, p=p, cohens_d=d)
    return df, stats_out


def experiment_F_protocols(N=60, E0=1.0, R_c=35.0, n_runs=5, max_rounds=3000):
    """Protocol comparison: EqOpt vs LEACH/PEGASIS/HEED, all multi-source."""
    rows = []
    rows += _run_many(N, E0, R_c, None, 'eqopt', n_runs, max_rounds)
    rows += _run_many(N, E0, R_c, None, 'static', n_runs, max_rounds)
    for name, fn in PROTOCOL_FUNCS.items():
        for r in range(n_runs):
            res = fn(N=N, E0=E0, max_rounds=max_rounds, seed=r, beta=0.5)
            res['N'] = N; res['E0'] = E0; res['strategy'] = name; res['seed'] = r
            rows.append(res)
    return pd.DataFrame(rows)
