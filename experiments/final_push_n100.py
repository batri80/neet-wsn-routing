"""
Targeted extension: eqopt_k vs one-step eqopt only (the closest-to-
significant comparison, p=0.059 at n=50), pushed to n=100 to properly
resolve whether this is a real, detectable effect.
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time
import signal

class TimeoutException(Exception):
    pass
def _handler(s, f):
    raise TimeoutException()

N, E0, R_c = 100, 1.0, 35.0
PSI_C, RC_THRESH, COOLDOWN, BETA = 0.04062, 0.51290, 30, 0.5  # beta=0.5 pending Concern 1 resolution
N_SEEDS = 100

results = {}
for strat, kwargs in [('eqopt', {}),
                       ('eqopt_k', dict(K=40, Psi_c=PSI_C, Rc_thresh=RC_THRESH,
                                         cooldown_rounds=COOLDOWN, beta=BETA))]:
    t0 = time.time()
    hnds = []
    for seed in range(N_SEEDS):
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(240)
        try:
            res = run_episode(N=N, E0=E0, R_c=R_c, strategy=strat, max_rounds=3000,
                               seed=seed, record_series=False, **kwargs)
            signal.alarm(0)
            hnds.append(res['HND'])
        except TimeoutException:
            pass
    results[strat] = np.array(hnds)
    print(f"{strat:10s}: mean={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
          f"n={len(hnds)}/{N_SEEDS}  time={time.time()-t0:.0f}s", flush=True)

a, b = results['eqopt_k'], results['eqopt']
u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
d = (a.mean() - b.mean()) / pooled_std
print(f"\neqopt_k vs eqopt (n=100): p={p:.4f}  d={d:.3f}  "
      f"{'SIGNIFICANT' if p<0.05 else 'not significant'}")
