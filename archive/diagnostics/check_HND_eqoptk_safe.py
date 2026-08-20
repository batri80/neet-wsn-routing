"""
Per-seed logged HND check with a hard per-episode timeout, so a
pathological seed is visible and skipped rather than hanging silently.
Also given the very high reconfiguration frequency we just observed
(seed 0: 2967 reconfigs in a 3000-round episode), the timeout here is
the primary safety mechanism -- the cascade-depth cap alone does not
bound how often projections are triggered over a full-length episode.
"""
from simulate import run_episode
import numpy as np
import time
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

N, E0, R_c = 100, 1.0, 35.0
MAX_ROUNDS = 3000
TIMEOUT_SEC = 180

hnds = []
for seed in range(20):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SEC)
    t0 = time.time()
    try:
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy='eqopt_k', K=40,
                           max_rounds=MAX_ROUNDS, seed=seed, record_series=False)
        signal.alarm(0)
        elapsed = time.time() - t0
        hnds.append(res['HND'])
        print(f"seed={seed:2d}: HND={res['HND']:5d}  n_reconfigs={res['n_reconfigs']:5d}  time={elapsed:.1f}s", flush=True)
    except TimeoutException:
        elapsed = time.time() - t0
        print(f"seed={seed:2d}: TIMED OUT after {elapsed:.1f}s -- skipping", flush=True)

print(f"\ncompleted seeds: {len(hnds)}/20", flush=True)
if hnds:
    print(f"mean HND (completed only): {np.mean(hnds):.1f}", flush=True)
