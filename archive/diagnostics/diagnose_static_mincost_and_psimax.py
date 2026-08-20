from simulate import run_episode
import numpy as np

print("=== reconfiguration behavior: static vs mincost vs eqopt, N=100 ===")
for strat in ['static', 'mincost', 'eqopt']:
    rows = [run_episode(N=100, E0=1.0, strategy=strat, max_rounds=3000, seed=s, record_series=False)
            for s in range(8)]
    n_local = np.mean([r['n_local'] for r in rows])
    n_global = np.mean([r['n_global'] for r in rows])
    tl = sum(r['total_with_alt_local'] for r in rows)
    hnd = np.mean([r['HND'] for r in rows])
    print(f"{strat:8s}: mean n_local={n_local:.1f}  mean n_global={n_global:.1f}  "
          f"local-triggers-with-a-feasible-alt={tl}  mean HND={hnd:.1f}")

print("\n=== Psi_max vs episode length -- is Psi_max just capped by early death? ===")
for strat in ['random', 'eqopt']:
    rows = [run_episode(N=500, E0=1.0, strategy=strat, max_rounds=3000, seed=s, record_series=False)
            for s in range(8)]
    for r in rows:
        print(f"  {strat:8s}: LND={r['LND']:5d}  Psi_max={r['Psi_max']:.4f}")

print("\n=== matched-window Psi (round 0-40, where even the shortest random episodes survive) ===")
for strat in ['random', 'eqopt']:
    vals = []
    for seed in range(10):
        res = run_episode(N=500, E0=1.0, strategy=strat, max_rounds=40, seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= 40:
            vals.append(psi[:40].max())
    print(f"{strat:8s}: mean Psi_max over rounds 0-40 (matched window) = {np.mean(vals):.5f}  "
          f"(n={len(vals)}/10 reached round 40)")
