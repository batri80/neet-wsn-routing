from simulate import run_episode
import numpy as np

N, E0, R_c = 60, 1.0, 35.0

# 1. Watch Psi(t) with reconfiguration effectively OFF, to see natural growth
print("--- Psi(t) growth with high threshold (rare reconfig) ---")
res = run_episode(N=N, E0=E0, R_c=R_c, strategy='eqopt', Psi_c=999,
                   Rc_thresh=999, max_rounds=500, seed=1, record_series=True)
psi_series = res['series']['Psi']
print("Psi at rounds 50,100,200,300,400:",
      [round(psi_series[min(r,len(psi_series)-1)], 5) for r in [50,100,200,300,400]])

# 2. Log R_h at every hole event across several seeds, to find empirical R_c
print("\n--- R_h at hole events (for R_c calibration) ---")
all_rh_at_holes = []
for seed in range(5):
    res = run_episode(N=N, E0=E0, R_c=R_c, strategy='random',
                       max_rounds=800, seed=seed, record_series=True)
    rh = res['series']['Rh']
    holes = rh[rh >= 0.5]  # loosen threshold to catch candidates
    all_rh_at_holes.extend(holes.tolist())
if all_rh_at_holes:
    print("10th percentile R_h among high-risk rounds:", np.percentile(all_rh_at_holes, 10))
    print("median:", np.percentile(all_rh_at_holes, 50))
