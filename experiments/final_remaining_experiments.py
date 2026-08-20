"""
final_remaining_experiments.py -- combines all four genuinely pending
experiments into one prioritized, resumable script:

  PRIORITY 1: EEUC full validation -- vs mincost-v3, static-CT, HEED,
              LEACH, full five-point N-sweep.
  PRIORITY 2: Experiment B extension -- energy sensitivity (E0=0.5,2.0)
              at N=200,500.
  PRIORITY 3: Experiment C extension -- sink topology (edge, corner)
              at N=200,500.
  PRIORITY 4: Blend experiment at N=500 completion -- alpha=0.5, 1.0.

Ordered by (value / cost): EEUC closes the last major baseline gap
cheaply; B/C extensions are moderate cost; the blend completion is
the most expensive and runs last.

Resumable: every result cached to disk immediately. Safe to interrupt
and rerun -- completed cells are skipped.

Run with caffeinate to prevent sleep interruption:
  caffeinate -i nohup python3 final_remaining_experiments.py > final_remaining_log.txt 2>&1 &
"""
import numpy as np
import json
import time
import os
from scipy import stats
from protocols import PROTOCOL_FUNCS
from protocol_eeuc import run_eeuc_episode
from flow_simulate_mincost import run_episode_v3_mincost
from flow_simulate_static_ct import run_episode_static_ct
from flow_simulate_blend import run_episode_v3_blend

OUTDIR = 'final_remaining_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
N_VALUES_FULL = [30, 60, 100, 200, 500]
N_SEEDS = 10


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def save_json(name, obj):
    with open(OUTDIR + '/' + name + '.json', 'w') as f:
        json.dump(obj, f, default=float)


def load_json(name):
    path = OUTDIR + '/' + name + '.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_hnds(name, compute_fn, prior_sources=None):
    cached = load_json(name)
    if cached is not None:
        return np.array(cached)
    if prior_sources:
        for src in prior_sources:
            if os.path.exists(src):
                with open(src) as f:
                    hnds = json.load(f)
                save_json(name, hnds)
                return np.array(hnds)
    hnds = compute_fn()
    save_json(name, hnds)
    return np.array(hnds)


log('=== PRIORITY 1: EEUC FULL VALIDATION ===')

eeuc_data, leach_data, heed_data, mincost_data, ct_data = {}, {}, {}, {}, {}

for N in N_VALUES_FULL:
    t0 = time.time()
    eeuc_data[N] = get_hnds(
        'eeuc_N' + str(N),
        lambda N=N: [run_eeuc_episode(N=N, E0=E0, R_c=R_c, p_candidate=0.05,
                                       max_rounds=MAX_ROUNDS, seed=s)['HND'] for s in range(N_SEEDS)])
    log('  EEUC N=' + str(N) + ': mean=' + str(round(eeuc_data[N].mean(), 1)) +
        '  time=' + str(round(time.time() - t0, 0)) + 's')

    leach_data[N] = get_hnds(
        'leach_N' + str(N),
        lambda N=N: [PROTOCOL_FUNCS['LEACH'](N=N, E0=E0, max_rounds=MAX_ROUNDS, seed=s, beta=1.0)['HND']
                     for s in range(N_SEEDS)],
        prior_sources=['redo_leach_results/LEACH_fixed_N' + str(N) + '.json'])

    heed_data[N] = get_hnds(
        'heed_N' + str(N),
        lambda N=N: [PROTOCOL_FUNCS['HEED'](N=N, E0=E0, max_rounds=MAX_ROUNDS, seed=s, beta=1.0)['HND']
                     for s in range(N_SEEDS)],
        prior_sources=['n_sweep_v3_vs_protocols_results/N' + str(N) + '_HEED.json'])

    ct_data[N] = get_hnds(
        'ct_N' + str(N),
        lambda N=N: [run_episode_static_ct(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                            max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
                     for s in range(N_SEEDS)],
        prior_sources=['redo_leach_results/ct_N' + str(N) + '.json'])

    mincost_data[N] = get_hnds(
        'mincost_N' + str(N),
        lambda N=N: [run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                             max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
                     for s in range(N_SEEDS)],
        prior_sources=['mincost_validation_results/N' + str(N) + '_v3_mincost.json',
                        'final_strategy_results/N' + str(N) + '_mincost_v3.json',
                        'redo_leach_results/mincost_v3_N' + str(N) + '.json'])

log('')
log('--- EEUC significance vs each baseline, per N ---')
for N in N_VALUES_FULL:
    eeuc = eeuc_data[N]
    for label, other in [('LEACH', leach_data[N]), ('HEED', heed_data[N]),
                          ('static-CT', ct_data[N]), ('mincost-v3', mincost_data[N])]:
        u, p = stats.mannwhitneyu(eeuc, other, alternative='two-sided')
        pooled = np.sqrt((eeuc.var(ddof=1) + other.var(ddof=1)) / 2)
        d = (eeuc.mean() - other.mean()) / pooled if pooled > 1e-12 else float('nan')
        sig = 'SIG' if p < 0.05 else 'n.s.'
        result = 'EEUC WINS' if eeuc.mean() > other.mean() else 'EEUC loses'
        log('  N=' + str(N) + ' EEUC(' + str(round(eeuc.mean(),1)) + ') vs ' + label +
            '(' + str(round(other.mean(),1)) + '): ' + result + '  p=' + str(round(p, 5)) +
            '  d=' + str(round(d, 3)) + '  ' + sig)

log('')
log('=== PRIORITY 2: EXPERIMENT B EXTENSION ===')

for N in [200, 500]:
    for E0_val in [0.5, 2.0]:
        results = {}
        for strat in ['mincost_v3', 'static_CT', 'LEACH', 'HEED']:
            name = 'B_N' + str(N) + '_E0_' + str(E0_val) + '_' + strat
            def compute(strat=strat, N=N, E0_val=E0_val):
                if strat == 'mincost_v3':
                    return [run_episode_v3_mincost(N=N, E0=E0_val, R_c=R_c, max_neighbors=15,
                                                     max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
                            for s in range(N_SEEDS)]
                elif strat == 'static_CT':
                    return [run_episode_static_ct(N=N, E0=E0_val, R_c=R_c, max_neighbors=15,
                                                    max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
                            for s in range(N_SEEDS)]
                else:
                    return [PROTOCOL_FUNCS[strat](N=N, E0=E0_val, max_rounds=MAX_ROUNDS, seed=s, beta=1.0)['HND']
                            for s in range(N_SEEDS)]
            t0 = time.time()
            results[strat] = get_hnds(name, compute)
            log('  N=' + str(N) + ' E0=' + str(E0_val) + ' ' + strat + ': mean=' +
                str(round(results[strat].mean(), 1)) + '  time=' + str(round(time.time() - t0, 0)) + 's')

log('')
log('=== PRIORITY 3: EXPERIMENT C EXTENSION ===')

SINK_POSITIONS = {'edge': [0.0, 50.0], 'corner': [0.0, 0.0]}
for N in [200, 500]:
    for sink_label, sink_pos in SINK_POSITIONS.items():
        results = {}
        for strat in ['mincost_v3', 'static_CT', 'LEACH', 'HEED']:
            name = 'C_N' + str(N) + '_' + sink_label + '_' + strat
            def compute(strat=strat, N=N, sink_pos=sink_pos):
                if strat == 'mincost_v3':
                    return [run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, sink_pos=sink_pos, max_neighbors=15,
                                                     max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
                            for s in range(N_SEEDS)]
                elif strat == 'static_CT':
                    return [run_episode_static_ct(N=N, E0=E0, R_c=R_c, sink_pos=sink_pos, max_neighbors=15,
                                                    max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
                            for s in range(N_SEEDS)]
                else:
                    return [PROTOCOL_FUNCS[strat](N=N, E0=E0, sink_pos=sink_pos,
                                                    max_rounds=MAX_ROUNDS, seed=s, beta=1.0)['HND']
                            for s in range(N_SEEDS)]
            t0 = time.time()
            results[strat] = get_hnds(name, compute)
            log('  N=' + str(N) + ' ' + sink_label + ' ' + strat + ': mean=' +
                str(round(results[strat].mean(), 1)) + '  time=' + str(round(time.time() - t0, 0)) + 's')

log('')
log('=== PRIORITY 4: BLEND N=500 COMPLETION ===')

for alpha in [0.0, 0.5, 1.0]:
    name = 'blend_N500_alpha' + str(alpha)
    cached = load_json(name)
    if cached:
        mean_hnd = cached.get('mean_hnd', np.mean(cached.get('hnds', [0])))
        log('  alpha=' + str(alpha) + ': [cached] mean_HND=' + str(round(mean_hnd, 1)))
        continue
    prior_path = 'overnight_results/blend_N500_alpha' + str(alpha) + '.json'
    if os.path.exists(prior_path):
        with open(prior_path) as f:
            prior = json.load(f)
        save_json(name, prior)
        log('  alpha=' + str(alpha) + ': [reused from ' + prior_path + '] mean_HND=' +
            str(round(prior['mean_hnd'], 1)))
        continue

    t0 = time.time()
    hnds, psimaxes = [], []
    for s in range(5):
        res = run_episode_v3_blend(N=500, E0=E0, R_c=R_c, alpha=alpha, max_neighbors=15,
                                    max_rounds=MAX_ROUNDS, seed=s, record_series=False)
        hnds.append(res['HND'])
        psimaxes.append(res['Psi_max'])
    result = dict(hnds=hnds, psimaxes=psimaxes, mean_hnd=float(np.mean(hnds)),
                  mean_psimax=float(np.mean(psimaxes)))
    save_json(name, result)
    log('  alpha=' + str(alpha) + ': mean_HND=' + str(round(result['mean_hnd'], 1)) +
        '  mean_Psi_max=' + str(round(result['mean_psimax'], 5)) +
        '  time=' + str(round(time.time() - t0, 0)) + 's')

log('')
log('--- Blend N=500 summary ---')
for alpha in [0.0, 0.5, 1.0]:
    r = load_json('blend_N500_alpha' + str(alpha))
    mean_hnd = r.get('mean_hnd', np.mean(r.get('hnds', [0])))
    mean_psimax = r.get('mean_psimax', np.mean(r.get('psimaxes', [0])))
    log('  alpha=' + str(alpha) + ': mean_HND=' + str(round(mean_hnd, 1)) +
        '  mean_Psi_max=' + str(round(mean_psimax, 5)))

log('')
log('=== ALL PRIORITIES COMPLETE ===')
log('=== DONE ===')
