"""
overnight_remaining_experiments.py -- combines the three genuinely
pending experiments (excluding the modern-baseline item, which
requires new protocol implementation, not just running existing code)
into one prioritized, resumable script:

  PRIORITY 1: Hole-risk verification -- does mincost-v3 reintroduce
              real energy-hole events at longer episodes? (safety/
              soundness question about the root-cause fix itself)
  PRIORITY 2: Blend experiment at N=500 -- does the "all-or-nothing"
              equilibrium-protection finding (N=100 only so far) hold
              at scale? SCOPED DOWN given demonstrated high cost
              (single blend episode at N=500 exceeded 280s) -- fewer
              alpha values, fewer seeds than the N=100 version.
  PRIORITY 3: Full N-sweep extension for Experiments B, C, D -- only
              N=100 tested so far; extends to N in {100, 200, 500}
              (three points, not the full five, to keep this
              genuinely overnight-tractable given demonstrated costs).

Resumable: every result cached to disk immediately. Safe to interrupt
(sleep, terminal close, etc.) and rerun -- completed cells are skipped.

Run with caffeinate to prevent sleep interruption:
  caffeinate -i nohup python3 overnight_remaining_experiments.py > overnight_log.txt 2>&1 &
"""
import numpy as np
import json
import time
import os
from scipy import stats
from flow_simulate_mincost import run_episode_v3_mincost
from flow_simulate_static_ct import run_episode_static_ct
from flow_simulate_blend import run_episode_v3_blend
from protocols import PROTOCOL_FUNCS

OUTDIR = 'overnight_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
LONG_ROUNDS = 6000  # for hole-risk verification at extended horizon


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def cache_or_compute(cache_name, compute_fn):
    cache_path = OUTDIR + '/' + cache_name + '.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            result = json.load(f)
        log('  [' + cache_name + '] [cached]')
        return result
    t0 = time.time()
    result = compute_fn()
    with open(cache_path, 'w') as f:
        json.dump(result, f)
    log('  [' + cache_name + '] computed  time=' + str(round(time.time() - t0, 0)) + 's')
    return result


# ======================================================================
# PRIORITY 1: Hole-risk verification
# ======================================================================
log('=== PRIORITY 1: HOLE-RISK VERIFICATION ===')

for N in [100, 500]:
    def compute_holes(N=N):
        hole_counts = []
        rh_means = []
        for s in range(10):
            res = run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                          max_rounds=LONG_ROUNDS, seed=s, record_series=True)
            rh_series = res['series']['Rh']
            n_hole_rounds = int((rh_series > 1.02488).sum())
            hole_counts.append(n_hole_rounds)
            rh_means.append(res['Rh_mean'])
        return dict(hole_counts=hole_counts, rh_means=rh_means,
                    mean_hole_rounds=float(np.mean(hole_counts)),
                    frac_episodes_with_holes=float(np.mean([h > 0 for h in hole_counts])))

    result = cache_or_compute('holerisk_N' + str(N), compute_holes)
    log('  N=' + str(N) + ': mean_hole_rounds=' + str(round(result['mean_hole_rounds'], 1)) +
        '  frac_episodes_with_any_hole=' + str(round(result['frac_episodes_with_holes'], 2)))

log('')

# ======================================================================
# PRIORITY 2: Blend experiment at N=500 (scoped down)
# ======================================================================
log('=== PRIORITY 2: BLEND AT N=500 (scoped) ===')

BLEND_ALPHAS_N500 = [0.0, 0.5, 1.0]
N_SEEDS_BLEND_N500 = 5

for alpha in BLEND_ALPHAS_N500:
    def compute_blend(alpha=alpha):
        hnds, psimaxes = [], []
        for s in range(N_SEEDS_BLEND_N500):
            res = run_episode_v3_blend(N=500, E0=E0, R_c=R_c, alpha=alpha,
                                        max_neighbors=15, max_rounds=MAX_ROUNDS,
                                        seed=s, record_series=False)
            hnds.append(res['HND'])
            psimaxes.append(res['Psi_max'])
        return dict(hnds=hnds, psimaxes=psimaxes,
                    mean_hnd=float(np.mean(hnds)), mean_psimax=float(np.mean(psimaxes)))

    result = cache_or_compute('blend_N500_alpha' + str(alpha), compute_blend)
    log('  alpha=' + str(alpha) + ': mean_HND=' + str(round(result['mean_hnd'], 1)) +
        '  mean_Psi_max=' + str(round(result['mean_psimax'], 5)))

log('')

# ======================================================================
# PRIORITY 3: Full N-sweep extension for Experiments B, C, D
# ======================================================================
log('=== PRIORITY 3: B/C/D EXTENDED N-SWEEP ===')

N_SEEDS_BCD = 8
N_VALUES_BCD = [100, 200, 500]

log('--- B: energy sensitivity, N=200,500 for E0=0.5,2.0 ---')
for N in [200, 500]:
    for E0_val in [0.5, 2.0]:
        def compute_b(N=N, E0_val=E0_val):
            results = {}
            for strat in ['mincost_v3', 'static_CT', 'LEACH', 'HEED']:
                hnds = []
                for s in range(N_SEEDS_BCD):
                    if strat == 'mincost_v3':
                        r = run_episode_v3_mincost(N=N, E0=E0_val, R_c=R_c, max_neighbors=15,
                                                    max_rounds=MAX_ROUNDS, seed=s, record_series=False)
                    elif strat == 'static_CT':
                        r = run_episode_static_ct(N=N, E0=E0_val, R_c=R_c, max_neighbors=15,
                                                   max_rounds=MAX_ROUNDS, seed=s, record_series=False)
                    else:
                        r = PROTOCOL_FUNCS[strat](N=N, E0=E0_val, max_rounds=MAX_ROUNDS, seed=s, beta=1.0)
                    hnds.append(r['HND'])
                results[strat] = hnds
            return results

        result = cache_or_compute('B_N' + str(N) + '_E0_' + str(E0_val), compute_b)
        means = {k: round(np.mean(v), 1) for k, v in result.items()}
        log('  N=' + str(N) + ' E0=' + str(E0_val) + ': ' + str(means))

log('')
log('--- C: sink topology, N=200,500 for edge,corner ---')
SINK_POSITIONS = {'edge': [0.0, 50.0], 'corner': [0.0, 0.0]}
for N in [200, 500]:
    for sink_label, sink_pos in SINK_POSITIONS.items():
        def compute_c(N=N, sink_pos=sink_pos):
            results = {}
            for strat in ['mincost_v3', 'static_CT', 'LEACH', 'HEED']:
                hnds = []
                for s in range(N_SEEDS_BCD):
                    if strat == 'mincost_v3':
                        r = run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, sink_pos=sink_pos,
                                                    max_neighbors=15, max_rounds=MAX_ROUNDS,
                                                    seed=s, record_series=False)
                    elif strat == 'static_CT':
                        r = run_episode_static_ct(N=N, E0=E0, R_c=R_c, sink_pos=sink_pos,
                                                   max_neighbors=15, max_rounds=MAX_ROUNDS,
                                                   seed=s, record_series=False)
                    else:
                        r = PROTOCOL_FUNCS[strat](N=N, E0=E0, sink_pos=sink_pos,
                                                   max_rounds=MAX_ROUNDS, seed=s, beta=1.0)
                    hnds.append(r['HND'])
                results[strat] = hnds
            return results

        result = cache_or_compute('C_N' + str(N) + '_' + sink_label, compute_c)
        means = {k: round(np.mean(v), 1) for k, v in result.items()}
        log('  N=' + str(N) + ' ' + sink_label + ': ' + str(means))

log('')
log('--- D: R_h correlation, N=500 (only N=100 tested so far) ---')
def compute_d(N=500):
    data = []
    for s in range(N_SEEDS_BCD):
        res = run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                      max_rounds=MAX_ROUNDS, seed=s, record_series=True)
        rh_series = res['series']['Rh']
        psi_series = res['series']['Psi']
        data.append(dict(HND=res['HND'],
                          Rh_at_50=float(rh_series[50]) if len(rh_series) > 50 else None,
                          Psi_at_50=float(psi_series[50]) if len(psi_series) > 50 else None))
    return data

result_d = cache_or_compute('D_N500', compute_d)
HND_d = np.array([d['HND'] for d in result_d])
Rh50_d = np.array([d['Rh_at_50'] for d in result_d if d['Rh_at_50'] is not None])
Psi50_d = np.array([d['Psi_at_50'] for d in result_d if d['Psi_at_50'] is not None])
if len(Rh50_d) == len(HND_d):
    rho_rh, p_rh = stats.spearmanr(Rh50_d, HND_d)
    rho_psi, p_psi = stats.spearmanr(Psi50_d, HND_d)
    log('  N=500: Rh@50 vs HND rho=' + str(round(rho_rh, 3)) + ' p=' + str(round(p_rh, 4)) +
        '  |  Psi@50 vs HND rho=' + str(round(rho_psi, 3)) + ' p=' + str(round(p_psi, 4)))

log('')
log('=== ALL PRIORITIES COMPLETE ===')
log('=== DONE ===')
