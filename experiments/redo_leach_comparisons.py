"""
redo_leach_comparisons.py -- comprehensive re-verification of every
LEACH-involving comparison in the model reference document, following
discovery of a genuine implementation bug: the CH-election threshold
formula correctly escalates toward 1.0 near the end of each epoch
(intended LEACH behavior), but without excluding nodes that already
served as CH earlier in the SAME epoch, this caused every node to
become a CH simultaneously in late-epoch rounds -- a degenerate,
cost-skipping state that inflated LEACH's measured performance.

Bug is FIXED directly in protocols.py (patch already applied). HEED
and PEGASIS were independently audited and confirmed NOT to have this
or an analogous bug.

This script recomputes ONLY LEACH (the invalidated piece) at the full
five-point N-sweep, then reuses all still-valid cached data (v3,
static-CT, HEED were never affected) to regenerate every downstream
comparison table and significance test.

Resumable: every result cached to disk immediately. Safe to interrupt
and rerun -- completed cells are skipped.

Run with caffeinate to prevent sleep interruption:
  caffeinate -i nohup python3 redo_leach_comparisons.py > redo_leach_log.txt 2>&1 &
"""
import numpy as np
import json
import time
import os
from scipy import stats
from protocols import PROTOCOL_FUNCS
from flow_simulate_mincost import run_episode_v3_mincost
from flow_simulate_static_ct import run_episode_static_ct

OUTDIR = 'redo_leach_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
N_VALUES = [30, 60, 100, 200, 500]
N_SEEDS = 15


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


# ======================================================================
# STEP 1: recompute LEACH (fixed) at all five N values
# ======================================================================
log('=== STEP 1: LEACH (fixed), full five-point sweep ===')
leach_fixed = {}
for N in N_VALUES:
    cache_name = 'LEACH_fixed_N' + str(N)
    cached = load_json(cache_name)
    if cached:
        leach_fixed[N] = np.array(cached)
        log('  N=' + str(N) + ': [cached] mean=' + str(round(leach_fixed[N].mean(), 1)))
        continue
    t0 = time.time()
    hnds = [PROTOCOL_FUNCS['LEACH'](N=N, E0=E0, max_rounds=MAX_ROUNDS, seed=s, beta=1.0)['HND']
            for s in range(N_SEEDS)]
    save_json(cache_name, hnds)
    leach_fixed[N] = np.array(hnds)
    log('  N=' + str(N) + ': mean=' + str(round(leach_fixed[N].mean(), 1)) +
        '  std=' + str(round(leach_fixed[N].std(), 1)) + '  time=' + str(round(time.time() - t0, 0)) + 's')

log('')
log('=== reference: original (buggy) LEACH values ===')
LEACH_ORIGINAL = {30: 587.9, 60: 691.1, 100: 755.5, 200: 814.9, 500: 845.8}
for N in N_VALUES:
    change_pct = 100 * (leach_fixed[N].mean() - LEACH_ORIGINAL[N]) / LEACH_ORIGINAL[N]
    log('  N=' + str(N) + ': original=' + str(LEACH_ORIGINAL[N]) + '  fixed=' +
        str(round(leach_fixed[N].mean(), 1)) + '  change=' + str(round(change_pct, 1)) + '%')

log('')

# ======================================================================
# STEP 2: mincost-v3 vs LEACH(fixed), full five-point sweep
# ======================================================================
log('=== STEP 2: mincost-v3 vs LEACH(fixed), full sweep ===')

PRIOR_MINCOST_SOURCES = [
    'mincost_validation_results/N{N}_v3_mincost.json',
    'final_strategy_results/N{N}_mincost_v3.json',
]

mincost_data = {}
for N in N_VALUES:
    cache_name = 'mincost_v3_N' + str(N)
    cached = load_json(cache_name)
    if cached:
        mincost_data[N] = np.array(cached)
        log('  mincost-v3 N=' + str(N) + ': [cached] mean=' + str(round(mincost_data[N].mean(), 1)))
        continue

    found = False
    for src_template in PRIOR_MINCOST_SOURCES:
        src_path = src_template.format(N=N)
        if os.path.exists(src_path):
            with open(src_path) as f:
                hnds = json.load(f)
            mincost_data[N] = np.array(hnds)
            save_json(cache_name, hnds)
            log('  mincost-v3 N=' + str(N) + ': [reused from ' + src_path + '] mean=' +
                str(round(mincost_data[N].mean(), 1)))
            found = True
            break
    if found:
        continue

    t0 = time.time()
    hnds = [run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                    max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
            for s in range(N_SEEDS)]
    save_json(cache_name, hnds)
    mincost_data[N] = np.array(hnds)
    log('  mincost-v3 N=' + str(N) + ': [computed fresh] mean=' + str(round(mincost_data[N].mean(), 1)) +
        '  time=' + str(round(time.time() - t0, 0)) + 's')

log('')
log('--- Updated significance: mincost-v3 vs LEACH(fixed) ---')
summary_rows = []
for N in N_VALUES:
    mc = mincost_data[N]
    lf = leach_fixed[N]
    u, p = stats.mannwhitneyu(mc, lf, alternative='two-sided')
    d = (mc.mean() - lf.mean()) / np.sqrt((mc.var(ddof=1) + lf.var(ddof=1)) / 2)
    sig = 'SIG' if p < 0.05 else 'n.s.'
    result = 'v3 WINS' if mc.mean() > lf.mean() else 'v3 loses'
    ratio = mc.mean() / lf.mean()
    summary_rows.append((N, mc.mean(), lf.mean(), ratio))
    log('  N=' + str(N) + ': mincost-v3=' + str(round(mc.mean(), 1)) + '  LEACH(fixed)=' +
        str(round(lf.mean(), 1)) + '  ratio=' + str(round(ratio, 3)) + '  ' + result +
        '  p=' + str(round(p, 5)) + '  d=' + str(round(d, 3)) + '  ' + sig)

Ns = [r[0] for r in summary_rows]
ratios = [r[3] for r in summary_rows]
rho, p_trend = stats.spearmanr(Ns, ratios)
log('  TREND: Spearman rho=' + str(round(rho, 3)) + '  p=' + str(round(p_trend, 5)) +
    '  ' + ('SIGNIFICANT' if p_trend < 0.05 else 'not significant'))

log('')

# ======================================================================
# STEP 3: frozen NEET-v3 (min-max fairness) vs LEACH(fixed)
# ======================================================================
log('=== STEP 3: frozen NEET-v3 (fairness) vs LEACH(fixed), full sweep ===')

frozen_v3_data = {}
for N in N_VALUES:
    cache_name = 'frozen_v3_N' + str(N)
    cached = load_json(cache_name)
    if cached:
        frozen_v3_data[N] = np.array(cached)
        log('  frozen-v3 N=' + str(N) + ': [cached] mean=' + str(round(frozen_v3_data[N].mean(), 1)))
        continue
    src_path = 'n_sweep_v3_vs_protocols_results/N' + str(N) + '_v3_flow.json'
    if os.path.exists(src_path):
        with open(src_path) as f:
            hnds = json.load(f)
        frozen_v3_data[N] = np.array(hnds)
        save_json(cache_name, hnds)
        log('  frozen-v3 N=' + str(N) + ': [reused from ' + src_path + '] mean=' +
            str(round(frozen_v3_data[N].mean(), 1)))
        continue
    from flow_simulate import run_episode_v3
    t0 = time.time()
    hnds = [run_episode_v3(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                            max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
            for s in range(N_SEEDS)]
    save_json(cache_name, hnds)
    frozen_v3_data[N] = np.array(hnds)
    log('  frozen-v3 N=' + str(N) + ': [computed fresh] mean=' + str(round(frozen_v3_data[N].mean(), 1)) +
        '  time=' + str(round(time.time() - t0, 0)) + 's')

log('')
log('--- Updated significance: frozen NEET-v3 vs LEACH(fixed) ---')
summary_rows_frozen = []
for N in N_VALUES:
    fv = frozen_v3_data[N]
    lf = leach_fixed[N]
    u, p = stats.mannwhitneyu(fv, lf, alternative='two-sided')
    d = (fv.mean() - lf.mean()) / np.sqrt((fv.var(ddof=1) + lf.var(ddof=1)) / 2)
    sig = 'SIG' if p < 0.05 else 'n.s.'
    result = 'v3 WINS' if fv.mean() > lf.mean() else 'v3 loses'
    ratio = fv.mean() / lf.mean()
    summary_rows_frozen.append((N, fv.mean(), lf.mean(), ratio))
    log('  N=' + str(N) + ': frozen-v3=' + str(round(fv.mean(), 1)) + '  LEACH(fixed)=' +
        str(round(lf.mean(), 1)) + '  ratio=' + str(round(ratio, 3)) + '  ' + result +
        '  p=' + str(round(p, 5)) + '  d=' + str(round(d, 3)) + '  ' + sig)

Ns_f = [r[0] for r in summary_rows_frozen]
ratios_f = [r[3] for r in summary_rows_frozen]
rho_f, p_trend_f = stats.spearmanr(Ns_f, ratios_f)
log('  TREND: Spearman rho=' + str(round(rho_f, 3)) + '  p=' + str(round(p_trend_f, 5)) +
    '  ' + ('SIGNIFICANT' if p_trend_f < 0.05 else 'not significant'))

log('')

# ======================================================================
# STEP 4: static-CT vs LEACH(fixed)
# ======================================================================
log('=== STEP 4: static-CT vs LEACH(fixed) ===')
ct_data = {}
for N in N_VALUES:
    cache_name = 'ct_N' + str(N)
    cached = load_json(cache_name)
    if cached:
        ct_data[N] = np.array(cached)
        continue
    t0 = time.time()
    hnds = [run_episode_static_ct(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                   max_rounds=MAX_ROUNDS, seed=s, record_series=False)['HND']
            for s in range(N_SEEDS)]
    save_json(cache_name, hnds)
    ct_data[N] = np.array(hnds)
    log('  static-CT N=' + str(N) + ': computed  mean=' + str(round(ct_data[N].mean(), 1)) +
        '  time=' + str(round(time.time() - t0, 0)) + 's')

for N in N_VALUES:
    ct = ct_data[N]
    lf = leach_fixed[N]
    u, p = stats.mannwhitneyu(ct, lf, alternative='two-sided')
    d = (ct.mean() - lf.mean()) / np.sqrt((ct.var(ddof=1) + lf.var(ddof=1)) / 2)
    sig = 'SIG' if p < 0.05 else 'n.s.'
    result = 'CT WINS' if ct.mean() > lf.mean() else 'CT loses'
    log('  N=' + str(N) + ': static-CT=' + str(round(ct.mean(), 1)) + '  LEACH(fixed)=' +
        str(round(lf.mean(), 1)) + '  ' + result + '  p=' + str(round(p, 5)) + '  d=' + str(round(d, 3)) + '  ' + sig)

log('')
log('=== ALL STEPS COMPLETE ===')
log('=== DONE ===')
