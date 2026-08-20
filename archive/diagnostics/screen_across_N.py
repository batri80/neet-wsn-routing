"""
Quick, cheap screening test: does eqopt_k's HND effect (vs one-step
eqopt and static) vary with N? Motivated by one-step EqOpt's documented
problem WORSENING with N (Section 14.6) -- if K-step helps at all, it
should show up most clearly at large N, not N=100 where we already
tested and found no significant effect.

NOT a final validation -- small n=12, reuses N=100-derived calibration
constants as a first-pass approximation. Purpose is to detect whether
ANY promising trend exists before committing to a full recalibration
and full-powered test at a specific N.
"""
import numpy as np
import json
import time
import signal
import os
from simulate import run_episode

OUTDIR = 'n_screen_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c_comm = 1.0, 35.0
K = 40
PSI_C, RC_THRESH, COOLDOWN, BETA = 0.04062, 0.51290, 30, 0.5
N_VALUES = [30, 200, 500]
N_SEEDS = 12
EPISODE_TIMEOUT = 300  # generous, since we don't yet know N=500's real cost


class TimeoutException(Exception):
    pass

def _handler(s, f):
    raise TimeoutException()

def run_with_timeout(**kwargs):
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(EPISODE_TIMEOUT)
    try:
        res = run_episode(**kwargs)
        signal.alarm(0)
        return res
    except TimeoutException:
        return None

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --- Step 0: profile ONE episode per N, per strategy, before committing ---
log("=== PROFILING: one episode per (N, strategy) before full sweep ===")
profile = {}
for N in N_VALUES:
    for strat, kwargs in [('static', {}), ('eqopt', {}),
                           ('eqopt_k', dict(K=K, Psi_c=PSI_C, Rc_thresh=RC_THRESH,
                                              cooldown_rounds=COOLDOWN, beta=BETA))]:
        t0 = time.time()
        res = run_with_timeout(N=N, E0=E0, R_c=R_c_comm, strategy=strat,
                                max_rounds=3000, seed=0, record_series=False, **kwargs)
        elapsed = time.time() - t0
        status = 'OK' if res is not None else 'TIMED OUT'
        hnd = res['HND'] if res else None
        profile[f'{N}_{strat}'] = elapsed
        log(f"  N={N:4d} {strat:10s}: {status}  time={elapsed:.1f}s  HND={hnd}")

total_estimated = sum(profile.values()) * N_SEEDS
log(f"\nEstimated total sweep time: {total_estimated/60:.1f} minutes "
    f"(based on single-episode profiling x {N_SEEDS} seeds)")
save_path = f'{OUTDIR}/profile.json'
with open(save_path, 'w') as f:
    json.dump(profile, f, indent=2)

if total_estimated > 3600 * 4:
    log("WARNING: estimated time exceeds 4 hours. Consider reducing N_SEEDS "
        "or N_VALUES before proceeding. Continuing anyway in 30s "
        "(Ctrl+C to abort).")
    time.sleep(30)

# --- Step 1: full screening sweep ---
log("\n=== SCREENING SWEEP ===")
results = {}
for N in N_VALUES:
    results[N] = {}
    for strat, kwargs in [('static', {}), ('eqopt', {}),
                           ('eqopt_k', dict(K=K, Psi_c=PSI_C, Rc_thresh=RC_THRESH,
                                              cooldown_rounds=COOLDOWN, beta=BETA))]:
        cache_key = f'{N}_{strat}'
        cache_path = f'{OUTDIR}/{cache_key}.json'
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                hnds = json.load(f)
            log(f"  N={N:4d} {strat:10s}: [cached] mean={np.mean(hnds):.1f}  n={len(hnds)}")
            results[N][strat] = hnds
            continue

        t0 = time.time()
        hnds, timeouts = [], 0
        for seed in range(N_SEEDS):
            res = run_with_timeout(N=N, E0=E0, R_c=R_c_comm, strategy=strat,
                                    max_rounds=3000, seed=seed, record_series=False, **kwargs)
            if res is None:
                timeouts += 1
                continue
            hnds.append(res['HND'])
        elapsed = time.time() - t0
        with open(cache_path, 'w') as f:
            json.dump(hnds, f)
        results[N][strat] = hnds
        log(f"  N={N:4d} {strat:10s}: mean={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
            f"n={len(hnds)}/{N_SEEDS}  timeouts={timeouts}  time={elapsed:.0f}s")

# --- Step 2: summary -- does the eqopt_k advantage (or lack thereof) trend with N? ---
log("\n=== SUMMARY: eqopt_k advantage over baselines, by N ===")
log(f"{'N':>5} {'eqopt_k':>10} {'eqopt':>10} {'static':>10} {'ek-eqopt':>10} {'ek-static':>10}")
for N in N_VALUES:
    ek = np.mean(results[N]['eqopt_k'])
    eq = np.mean(results[N]['eqopt'])
    st = np.mean(results[N]['static'])
    log(f"{N:5d} {ek:10.1f} {eq:10.1f} {st:10.1f} {ek-eq:10.1f} {ek-st:10.1f}")

log("\nDone. If ek-eqopt or ek-static grows (more positive) with N, that's a "
    "promising trend worth a full-powered follow-up at the most promising N. "
    "If flat or negative across N, likely confirms the N=100 null result "
    "generalizes and this line of investigation should close.")
