"""
run_all.py -- runs Experiments A-F end to end at a reduced scale suitable
for validation / a quick pass, saves result tables (CSV) and figures (PNG).

Scale up via the constants below (N_RUNS, MAX_ROUNDS, N-lists) for a full
research-grade pass -- see README.md for guidance and expected runtimes.
"""
import os
import time
import json
import pandas as pd
from simulate import run_episode
import experiments as E
import plotting as P

RESULTS_DIR = 'results'
FIG_DIR = 'figures'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Reduced-scale defaults for a fast validation pass. Original paper used
# Nruns=15-30; scale these up for a real experimental campaign.
N_RUNS = 15
MAX_ROUNDS = 3000


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    t0 = time.time()

    log('Experiment A: scaling')
    dfA = E.experiment_A_scaling(Ns=(30, 60, 100), n_runs=N_RUNS, max_rounds=MAX_ROUNDS)
    dfA.to_csv(f'{RESULTS_DIR}/exp_A_scaling.csv', index=False)
    P.plot_scaling(dfA, f'{FIG_DIR}/exp_A_scaling.png')

    log('Experiment B: energy sensitivity')
    dfB = E.experiment_B_energy(E0s=(0.5, 1.0, 2.0), n_runs=N_RUNS, max_rounds=MAX_ROUNDS)
    dfB.to_csv(f'{RESULTS_DIR}/exp_B_energy.csv', index=False)
    P.plot_energy_sensitivity(dfB, f'{FIG_DIR}/exp_B_energy.png')

    log('Experiment C: sink topology')
    dfC = E.experiment_C_topology(n_runs=N_RUNS, max_rounds=MAX_ROUNDS)
    dfC.to_csv(f'{RESULTS_DIR}/exp_C_topology.csv', index=False)
    P.plot_topology(dfC, f'{FIG_DIR}/exp_C_topology.png')

    log('Experiment D: correlation analysis')
    dfD, corrD = E.experiment_D_correlation(Ns=(30, 60, 100), n_runs=N_RUNS, max_rounds=MAX_ROUNDS)
    dfD.to_csv(f'{RESULTS_DIR}/exp_D_correlation.csv', index=False)
    with open(f'{RESULTS_DIR}/exp_D_correlation_stats.json', 'w') as f:
        json.dump(corrD, f, indent=2, default=float)

    log('Experiment E: ablation')
    dfE, statsE = E.experiment_E_ablation(n_runs=max(N_RUNS, 6), max_rounds=MAX_ROUNDS)
    dfE.to_csv(f'{RESULTS_DIR}/exp_E_ablation.csv', index=False)
    with open(f'{RESULTS_DIR}/exp_E_ablation_stats.json', 'w') as f:
        json.dump(statsE, f, indent=2, default=float)
    P.plot_ablation(dfE, f'{FIG_DIR}/exp_E_ablation.png')

    log('Experiment F: protocol comparison')
    dfF = E.experiment_F_protocols(n_runs=N_RUNS, max_rounds=MAX_ROUNDS)
    dfF.to_csv(f'{RESULTS_DIR}/exp_F_protocols.csv', index=False)
    P.plot_protocol_comparison(dfF, f'{FIG_DIR}/exp_F_protocols.png')

    log('Sample time-series figure (single EqOpt episode)')
    res = run_episode(N=60, E0=1.0, R_c=35.0, strategy='eqopt',
                       max_rounds=MAX_ROUNDS, seed=42, record_series=True)
    P.plot_time_series(res['series'], f'{FIG_DIR}/sample_eqopt_timeseries.png')

    # Summary tables (Table III / IV analog)
    summary_rows = []
    for name, df in [('A', dfA), ('B', dfB), ('E', dfE), ('F', dfF)]:
        g = df.groupby('strategy').agg(
            AUC_Psi_mean=('AUC_Psi', 'mean'), AUC_Psi_std=('AUC_Psi', 'std'),
            HND_mean=('HND', 'mean'), HND_std=('HND', 'std'),
            hole_events_mean=('hole_events', 'mean')).reset_index()
        g['experiment'] = name
        summary_rows.append(g)
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv(f'{RESULTS_DIR}/summary_table.csv', index=False)

    log(f'Done in {time.time()-t0:.1f}s. Results -> {RESULTS_DIR}/, figures -> {FIG_DIR}/')


if __name__ == '__main__':
    main()
